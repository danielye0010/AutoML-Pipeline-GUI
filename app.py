from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import re
import subprocess
import threading
from pathlib import Path
import pandas as pd
import zipfile
from autogluon.tabular import TabularPredictor
import sweetviz as sv
import uuid
from threading import Lock

app = Flask(__name__)

REPO_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = Path(os.getenv("KANE_DATA_DIR", REPO_ROOT / "kaggle_data")).resolve()
WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
COMPETITION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# Thread-safe dictionary to store progress
progress_data = {}
progress_lock = Lock()


def validate_competition_name(competition_name):
    """Validate a Kaggle competition slug before using it in CLI/path operations."""
    if not competition_name or not COMPETITION_RE.fullmatch(competition_name):
        raise ValueError(
            "Competition name must be a Kaggle slug containing only letters, numbers, '.', '_' or '-'."
        )
    return competition_name


def competition_workspace(competition_name):
    competition_name = validate_competition_name(competition_name)
    path = (WORKSPACE_ROOT / competition_name).resolve()
    if os.path.commonpath([str(WORKSPACE_ROOT), str(path)]) != str(WORKSPACE_ROOT):
        raise ValueError("Competition path must remain inside the KANE workspace.")
    return path


def safe_extract_zip(zip_path, destination):
    """Extract a ZIP archive without allowing members to escape the destination."""
    destination = Path(destination).resolve()
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for member in zip_ref.infolist():
            member_path = (destination / member.filename).resolve()
            if os.path.commonpath([str(destination), str(member_path)]) != str(destination):
                raise ValueError(f"Unsafe archive member path: {member.filename}")
        zip_ref.extractall(destination)


def download_data(competition_name, download_path):
    competition_name = validate_competition_name(competition_name)
    download_path = Path(download_path)
    download_path.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        ["kaggle", "competitions", "download", "-c", competition_name, "-p", str(download_path)],
        check=True,
    )

    for file in download_path.iterdir():
        if file.suffix.lower() == ".zip":
            safe_extract_zip(file, download_path)
            file.unlink()


def detect_problem_type(train_data, label_column):
    unique_values = train_data[label_column].nunique()
    dtype = train_data[label_column].dtype
    if pd.api.types.is_numeric_dtype(dtype):
        if unique_values <= 2:
            return "binary"
        elif unique_values <= 20:
            return "multiclass"
        else:
            return "regression"
    else:
        return "multiclass" if unique_values > 2 else "binary"


def get_eval_metrics():
    return [
        "auto",
        "accuracy",
        "roc_auc",
        "log_loss",
        "f1",
        "precision",
        "recall",
        "rmse",
        "mse",
        "mae",
        "r2",
    ]


def train_and_predict(train_data, test_data, label_column, problem_type, eval_metric, time_limit, preset):
    if problem_type == "auto":
        problem_type = detect_problem_type(train_data, label_column)

    predictor = TabularPredictor(label=label_column, problem_type=problem_type, eval_metric=eval_metric)
    predictor.fit(train_data, presets=preset, time_limit=time_limit)
    predictions = predictor.predict(test_data)
    return predictions


def run_training(competition_name, label_column, problem_type, eval_metric, time_limit, id_column, preset, task_id):
    try:
        with progress_lock:
            progress_data[task_id] = 0

        # Download data into the repository-local KANE workspace.
        download_path = competition_workspace(competition_name)
        download_data(competition_name, download_path)
        with progress_lock:
            progress_data[task_id] = 10

        train_file = download_path / "train.csv"
        test_file = download_path / "test.csv"

        if not train_file.exists() or not test_file.exists():
            raise FileNotFoundError(f"Could not find 'train.csv' or 'test.csv' in {download_path}.")

        # Read data
        train = pd.read_csv(train_file)
        test = pd.read_csv(test_file)
        if label_column not in train.columns:
            raise KeyError(f"Label column '{label_column}' was not found in train.csv.")
        if id_column not in test.columns:
            raise KeyError(f"ID column '{id_column}' was not found in test.csv.")
        test_ids = test[id_column]
        with progress_lock:
            progress_data[task_id] = 20

        # Prepare data
        test_data_for_training = test.copy()
        with progress_lock:
            progress_data[task_id] = 30

        # Detect problem type if needed
        if problem_type == "auto":
            problem_type = detect_problem_type(train, label_column)
        with progress_lock:
            progress_data[task_id] = 40

        # Start training. Keep each run isolated inside its competition workspace.
        model_path = download_path / "models" / task_id
        predictor = TabularPredictor(
            label=label_column,
            problem_type=problem_type,
            eval_metric=eval_metric,
            path=str(model_path),
        )
        predictor.fit(train, presets=preset, time_limit=time_limit)
        with progress_lock:
            progress_data[task_id] = 80

        # Make predictions
        predictions = predictor.predict(test_data_for_training)
        with progress_lock:
            progress_data[task_id] = 90

        # Save submission file
        submission = pd.DataFrame({id_column: test_ids, label_column: predictions})
        submission_file = download_path / f"{competition_name}_submission.csv"
        submission.to_csv(submission_file, index=False)
        with progress_lock:
            progress_data[task_id] = 100

    except Exception as e:
        with progress_lock:
            progress_data[task_id] = "error"
            progress_data[task_id + "_error"] = str(e)


@app.route("/")
def index():
    eval_metrics = get_eval_metrics()
    time_limit_options = ["5 Minutes", "30 Minutes", "1 Hour", "2 Hours", "10 Hours"]
    presets = ["best_quality", "good_quality", "medium_quality"]
    problem_types = ["auto", "binary", "multiclass", "regression"]
    return render_template(
        "index.html",
        eval_metrics=eval_metrics,
        time_limit_options=time_limit_options,
        presets=presets,
        problem_types=problem_types,
    )


@app.route("/start_training", methods=["POST"])
def start_training():
    competition_name = (request.form.get("competition_name") or "").strip()
    label_column = (request.form.get("label_column") or "").strip()
    id_column = (request.form.get("id_column") or "").strip() or "Id"
    problem_type = request.form.get("problem_type") or "auto"
    eval_metric = request.form.get("eval_metric") or "auto"
    time_limit_str = request.form.get("time_limit")
    preset = request.form.get("preset") or "medium_quality"

    if not competition_name or not label_column:
        return "Please provide both the competition name and label column.", 400

    try:
        validate_competition_name(competition_name)
    except ValueError as exc:
        return str(exc), 400

    time_limit_options = {
        "5 Minutes": 300,
        "30 Minutes": 1800,
        "1 Hour": 3600,
        "2 Hours": 7200,
        "10 Hours": 36000,
    }
    time_limit = time_limit_options.get(time_limit_str, 3600)

    task_id = str(uuid.uuid4())
    with progress_lock:
        progress_data[task_id] = 0

    threading.Thread(
        target=run_training,
        args=(
            competition_name,
            label_column,
            problem_type,
            eval_metric if eval_metric != "auto" else None,
            time_limit,
            id_column,
            preset,
            task_id,
        ),
        daemon=True,
    ).start()

    return redirect(url_for("status", task_id=task_id))


@app.route("/status/<task_id>")
def status(task_id):
    return render_template("status.html", task_id=task_id)


@app.route("/progress/<task_id>")
def progress(task_id):
    with progress_lock:
        progress = progress_data.get(task_id, None)
        error = progress_data.get(task_id + "_error", None)
    if progress is None:
        return jsonify({"status": "unknown"})
    elif progress == "error":
        return jsonify({"status": "error", "error": error})
    else:
        return jsonify({"status": "progress", "progress": progress})


@app.route("/generate_eda_report", methods=["POST"])
def generate_eda_report():
    try:
        competition_name = (request.form.get("competition_name") or "").strip()
        download_path = competition_workspace(competition_name)
        train_file = download_path / "train.csv"

        if not train_file.exists():
            return "Training data not found. Please download data first.", 400

        train_data = pd.read_csv(train_file)
        report = sv.analyze(train_data)
        report_file = download_path / "sweetviz_report.html"
        report.show_html(str(report_file))

        return f"EDA report generated and saved at {report_file}."
    except Exception as e:
        return f"Failed to generate EDA report: {e}", 500


if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "0").lower() in {"1", "true", "yes"}
    app.run(debug=debug)
