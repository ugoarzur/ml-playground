# AI Guidelines — ML

> Extends root `AGENTS.md`. Rules here take precedence within this scope.

## Scope

All machine learning logic: data loading, preprocessing, feature engineering, model training, evaluation, and model export.

## Conventions

### Project Layout

Every project under `ml/projects/<name>/` uses the same folder structure:

- `data/` — raw and processed datasets
- `experiments/` — experiment scripts and exploratory work
- `features/` — feature engineering pipelines
- `models/` — model training scripts (one function per model type)
- `preprocessing/` — data cleaning and transformation logic
- `trained_models/` — serialized models (`.pkl`, `.joblib`)

When creating a new ML project, replicate this structure.

### Jupyter Notebooks (`ml/jupyter/`)

- Notebooks are a **presentation and execution layer** — they import and call code from `ml/projects/`.
- Do not write standalone ML logic in notebooks. Extract it into the appropriate project module first.

### Shared Utilities (`ml/shared/`)

- Code reusable across multiple projects (visualization helpers, common metrics, shared preprocessing steps) goes here.
- Keep helpers focused and small — one responsibility per module.

### ML Workflow

The standard pipeline for any project follows this order:

1. Load/fetch dataset → `data/`
2. Check for missing values, inspect types → `preprocessing/`
3. Clean data, handle nulls, scale features → `preprocessing/`
4. Engineer features → `features/`
5. Split into train/test sets
6. Train model → `models/`
7. Evaluate model (accuracy, r2, confusion matrix, etc.)
8. Export trained model → `trained_models/`
