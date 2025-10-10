# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A machine learning playground for experimenting with ML models and serving predictions via FastAPI. The project includes:
- ML experiments on Titanic survival prediction (KNeighborsClassifier) and California housing prices (LinearRegression)
- FastAPI backend to serve trained models
- Jupyter notebooks for exploratory data analysis and model training

## Package Management & Environment

This project uses **uv** as the package manager (not pip or poetry). All dependency commands must use uv:

```bash
# Install dependencies
uv sync

# Run Python scripts
uv run python <script.py>

# Start Jupyter Lab
uv sync && jupyter lab
```

Python version: >=3.12

## Development Commands

### Running the API Server
```bash
# Start FastAPI development server
uv run fastapi dev backend/main.py
```

### Working with Jupyter Notebooks
```bash
# Launch Jupyter Lab for ML experimentation
jupyter lab
```

Notebooks are located in `ml/jupyter/`:
- `titanic.ipynb` - Titanic survival prediction
- `house_pricing_guess.ipynb` - California housing price prediction

### Testing
```bash
# Run tests (pytest is in dev dependencies)
uv run pytest
```

## Architecture

The codebase follows a clear separation between ML experimentation and production serving:

### Directory Structure
```
ml-playground/
├── backend/              # FastAPI application
│   ├── main.py          # FastAPI app entry point
│   └── titanic.py       # Standalone Titanic ML pipeline (preprocess, train, evaluate)
├── ml/                  # ML code and experiments
│   ├── jupyter/         # Jupyter notebooks for experimentation
│   ├── projects/        # Organized by ML project
│   │   ├── titanic/     # (planned structure, not yet implemented)
│   │   └── houses/
│   │       ├── models/regression.py  # LinearRegression training function
│   │       └── trained_models/       # Serialized models (.pkl, .joblib)
│   └── shared/          # (planned) Shared utilities for visualization, metrics, preprocessing
└── assets/              # Datasets
    └── titanic/         # Titanic CSV data
```

### Key Architecture Patterns

1. **ML Project Structure**: Each ML project under `ml/projects/` should follow this pattern:
   - `data/` - Raw and processed datasets
   - `features/` - Feature engineering code
   - `models/` - Model training scripts
   - `trained_models/` - Serialized models (.pkl or .joblib files)

2. **Model Serialization**: Models are saved using `joblib.dump()` and loaded with `joblib.load()` (see [regression.py:8](ml/projects/houses/models/regression.py#L8) and [main.py:23](backend/main.py#L23))

3. **Data Preprocessing Workflow** (from README):
   - Fetch/load dataset
   - Check for missing values and correct data types (models require numeric input)
   - Prepare data: clean DataFrame, convert missing data to 0-1 scale using MinMaxScaler
   - Create features and target variables, split with `train_test_split()`
   - Train model
   - Evaluate model

4. **Backend Integration**: FastAPI loads trained models at startup and exposes prediction endpoints. Currently implemented for houses project at `/predict/houses`.

### Titanic Project Details

The Titanic project ([backend/titanic.py](backend/titanic.py)) is a complete ML pipeline in a single file:

- **Data preprocessing** ([preprocess_data:23-62](backend/titanic.py#L23-L62)):
  - Drops unused columns: PassengerId, Name, Ticket, Cabin, Embarked
  - Fills missing ages with median per passenger class (1st/2nd/3rd class have different age distributions)
  - Converts Sex to binary (male=1, female=0)
  - Feature engineering: FamilySize, IsAlone, FareBin, AgeBin

- **Model**: KNeighborsClassifier with GridSearchCV hyperparameter tuning
  - Tested parameters: n_neighbors (1-20), metric (euclidean/manhattan/minkowski), weights (uniform/distance)

- **Evaluation**: accuracy_score and confusion_matrix visualization with seaborn heatmap

### Houses Project

- Uses sklearn's `fetch_california_housing()` dataset (not from Kaggle)
- Model: LinearRegression ([ml/projects/houses/models/regression.py](ml/projects/houses/models/regression.py))
- Less structured than Titanic project, mostly experimental

## Data Sources

- **Titanic**: Kaggle dataset at `assets/titanic/titanic.csv` ([Kaggle link](https://www.kaggle.com/datasets/brendan45774/test-file))
- **Houses**: California housing data from `sklearn.datasets.fetch_california_housing()`

## Important Notes

- The backend/main.py references a houses prediction endpoint but it's incomplete (missing HouseFeatures model definition)
- The planned architecture in the README shows a more organized structure with shared utilities, but the current implementation has most code in standalone scripts
- When adding new ML projects, follow the houses project structure: `ml/projects/<project_name>/{data,features,models,trained_models}/`
