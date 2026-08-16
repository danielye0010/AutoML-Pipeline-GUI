# KANE — Kaggle AutoML No-Code Engine

KANE is a local web application for configuring and running Kaggle tabular AutoML workflows through a simple GUI. It combines the Kaggle CLI for competition data acquisition, AutoGluon for model training and ensembling, and Sweetviz for exploratory data analysis.

The goal is to make a standard competition workflow accessible from one interface: download data, choose the prediction setup, train within a time budget, generate predictions, and export a submission file.

## Features

- **No-code configuration** for competition name, label column, ID column, problem type, evaluation metric, time budget, and AutoGluon preset.
- **Automated Kaggle data acquisition** using the Kaggle CLI.
- **AutoML training** through `autogluon.tabular.TabularPredictor`.
- **Submission generation** from the competition test set.
- **EDA reports** using Sweetviz.
- **Progress/error reporting** through the Flask interface.
- **Isolated run directories** so model artifacts remain inside the competition workspace.

## Local architecture

```text
Browser
  ↓
Flask UI (`app.py`)
  ├── Kaggle CLI → competition data
  ├── AutoGluon → trained models / predictions
  └── Sweetviz → EDA report
```

KANE is designed as a **local development tool**, not as an internet-facing multi-user service.

## Installation

Create an isolated Python environment, then install the repository dependencies:

```bash
pip install -r requirements.txt
```

AutoGluon has platform- and Python-version-specific dependencies, so an isolated environment is recommended.

## Kaggle authentication

Configure the Kaggle CLI before launching KANE. Use your normal Kaggle API credential setup (for example, the Kaggle CLI credential file or supported environment variables). Credentials are not stored by KANE and should never be committed to this repository.

You can verify the CLI independently before starting the app:

```bash
kaggle competitions list
```

## Launch

```bash
python app.py
```

Then open the local Flask address shown in the terminal.

Flask debug mode is **off by default**. For local debugging only:

```bash
FLASK_DEBUG=1 python app.py
```

## Workspace

By default, competition data and model outputs are kept under:

```text
kaggle_data/<competition-slug>/
├── train.csv
├── test.csv
├── models/<task-id>/
├── <competition-slug>_submission.csv
└── sweetviz_report.html
```

Override the workspace root with:

```bash
export KANE_DATA_DIR=/path/to/kane-workspace
```

The web form accepts Kaggle-style competition slugs only. Download commands are executed with an argument list rather than shell interpolation, and extracted ZIP members are checked to remain inside the selected workspace.

## Training workflow

1. Enter the Kaggle competition slug.
2. Specify the target/label column and test-set ID column.
3. Select a problem type or let KANE infer a basic type from the target.
4. Choose an evaluation metric, time limit, and AutoGluon preset.
5. Start training and follow the status page.
6. Retrieve the generated submission CSV from the competition workspace.

## AutoML scope

KANE delegates model preprocessing, model selection, ensembling, and time-budgeted training to AutoGluon. The exact models available depend on the installed AutoGluon environment and optional dependencies.

The repository is a practical orchestration/UI project rather than a new AutoML algorithm: its value is integrating competition acquisition, configuration, training, reporting, and submission generation into one workflow.

## Other files

- `autonlp.py` — experimental NLP-oriented AutoML workflow.
- `ttk-version.py` — desktop/Tk interface experiment.
- `easy_test.py` and `titanic/` — lightweight example/test assets.

## License

KANE is released under the MIT License. See `LICENSE` for details.
