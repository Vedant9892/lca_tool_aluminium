# AI-Driven LCA Agent for Aluminium

The AI-Driven LCA Agent for Aluminium is a life cycle assessment platform for analysing aluminium products across manufacturing routes, energy sources, and end-of-life scenarios. It combines a Python-based analytics backend, machine learning training pipelines, and a browser-based dashboard to estimate environmental footprint, recycling potential, and cost-related impacts for batch-level product inputs.

Project materials describe the system as an AI agent for aluminium products that measures carbon resource usage, evaluates recycle potential, supports database-backed batch management, and provides real-time analytics with high predictive performance.

## Project Overview

This repository focuses on the operational assessment of aluminium product lifecycles. It loads raw production datasets, merges and normalises them into a processed training table, trains regression models for environmental and cost metrics, and serves a dashboard that computes stage-wise impact breakdowns for a selected product configuration.

The application is intended for analysts, sustainability teams, and product engineers who need a structured view of manufacturing emissions, energy consumption, material recovery potential, wastewater output, and related cost signals.

## Professional Project Description

The project implements a practical LCA workflow for aluminium sheet and pipe products. It ingests product and process data from multiple routes, standardises the feature space, and generates predictive models that support environmental impact estimation at both per-unit and batch levels. The web interface is built for direct parameter entry and visual review of computed stage outputs, while the Python training utilities support offline model development and experimentation.

## Key Features

- Stage-wise life cycle assessment for aluminium sheet and pipe workflows.
- Conventional and recycled route support, including bauxite-grade context.
- Energy-source and end-of-life scenario modelling.
- Batch-level and per-unit environmental breakdowns.
- Machine learning training pipeline for impact and cost prediction.
- Processed dataset generation from multiple raw production CSV files.
- Browser dashboard for operational analysis and result review.
- CLI helper for validating dashboard calculations from the terminal.

## Tech Stack / Technologies Used

- Python 3 for data processing, training, and API delivery.
- Flask for the web application and HTTP API.
- Pandas and NumPy for data preparation and numeric processing.
- scikit-learn for preprocessing and regression modelling.
- Joblib for model persistence.
- XGBoost and LightGBM as optional model candidates.
- HTML, CSS, and Vanilla JavaScript for the dashboard UI.
- Chart.js for dashboard visualisation.
- Flask-CORS for cross-origin support.

## File Structure

```text
lca_tool_aluminium/
├── api/
│   ├── main.py
│   └── readme.txt
├── data/
│   ├── featurizers/
│   │   ├── bauxite.py
│   │   ├── pipe.py
│   │   ├── recycle.py
│   │   └── sheet.py
│   ├── processed/
│   │   └── train.csv
│   ├── raw/
│   ├── raw2/
│   └── train_2.csv
├── scripts/
│   └── test_dashboard_cli.py
├── src/
│   ├── data/
│   │   ├── data_loader.py
│   │   └── merge.py
│   ├── inference/
│   │   ├── baselines.py
│   │   └── stage_engine.py
│   └── training/
│       ├── train_all.py
│       └── train_all_robust.py
├── web/
│   ├── app.js
│   ├── index.html
│   └── style.css
├── run.py
├── test.py
└── train_all.py
```

## Installation Instructions

### Prerequisites

- Python 3.10 or later.
- pip.
- A virtual environment tool such as `venv`.

### Create and Activate a Virtual Environment

From the project root, create a virtual environment and activate it.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux, use:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

Install the core packages used by the project:

```powershell
pip install flask flask-cors pandas numpy scikit-learn joblib
```

Install optional model packages if you want the full training model zoo:

```powershell
pip install xgboost lightgbm
```

If you work with additional file formats or extend the featurizers, you may also need packages such as `openpyxl` or `pyarrow`.

## How to Run the Project

### 1. Prepare the Data

Ensure the raw aluminium datasets are available under `data/raw/`. The project expects the source CSV files already present in that directory.

### 2. Generate the Processed Training Table

Run the training pipeline to merge the raw datasets and create the processed table:

```powershell
python train_all.py
```

This command writes the merged dataset to `data/processed/train.csv` and stores training outputs under generated model and experiment directories.

### 3. Start the Application Server

Launch the Flask application from the repository root:

```powershell
python api/main.py
```

You can also use the convenience launcher:

```powershell
python run.py
```

The dashboard is served locally at `http://localhost:5000` by default.

### 4. Open the Dashboard

Open `http://localhost:5000` in a browser to access the aluminium LCA interface.

## Environment Variables

The project does not include a committed `.env.example` file, but the backend code is configured to use environment-driven settings when external integrations are enabled.

| Variable | Purpose |
| --- | --- |
| `GROK_API_KEY` | Optional xAI key for external reasoning or analysis services. |
| `GROK_API_URL` | Endpoint for Grok-compatible analysis requests. |
| `GROK_MODEL` | Model identifier used by external inference services. |
| `YOLO_MODEL_PATH` | Path to local model weights if you extend the pipeline. |
| `CROWD_THRESHOLD` | Threshold value for classification logic in related analysis flows. |
| `SNAPSHOT_DIR` | Directory for saved evidence or result snapshots. |
| `SUPABASE_URL` | Supabase project URL for optional storage integration. |
| `SUPABASE_KEY` | Supabase access key for optional integration. |
| `MONGODB_URI` | MongoDB connection string if database storage is enabled. |
| `MONGODB_DB` | MongoDB database name. |
| `VIDEOS_DIR` | Directory for source or archived video assets. |
| `ACCIDENT_FRAMES_DIR` | Directory for incident frame captures. |
| `HOST` | Host address used by the Flask server. |
| `PORT` | Port used by the Flask server. |

## Usage Instructions

1. Enter the product type, batch size, manufacturing route, bauxite grade, energy source, and end-of-life option in the dashboard.
2. Provide the appropriate dimensions for pipe or sheet products.
3. Submit the configuration to calculate the life cycle impact breakdown.
4. Review the generated summary cards, charts, and stage-level metrics.
5. Use the CLI helper in `scripts/test_dashboard_cli.py` if you want to validate a scenario from the terminal.
6. Re-train the models when the raw datasets change or when you want to compare new model settings.

## Scripts / Commands

### Training and Analysis

| Command | Description |
| --- | --- |
| `python train_all.py` | Merge raw datasets, train models, and write processed outputs. |
| `python scripts/test_dashboard_cli.py --product sheet --units 100 --route conventional --grade high --energy renewable` | Execute a CLI validation scenario for the dashboard pipeline. |
| `python test.py` | Load a trained bundle and print sample predictions. |

### Application Server

| Command | Description |
| --- | --- |
| `python api/main.py` | Start the Flask application and dashboard server. |
| `python run.py` | Convenience launcher for the API entry point. |

## Contribution Guidelines

1. Create a dedicated branch for your changes.
2. Keep modifications focused and aligned with the existing data-science and Flask structure.
3. Update the documentation when you add a new dataset, model, route, or dashboard flow.
4. Re-run the training or validation scripts after changing data preparation logic.
5. Prefer small pull requests with clear summaries of the analytical or functional impact.

## License Information

No explicit license file is included in this repository. Add a license before public redistribution or external collaboration.