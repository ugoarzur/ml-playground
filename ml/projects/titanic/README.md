# Titanic — Survival Prediction

Binary classification predicting passenger survival on the Titanic, using a KNeighborsClassifier with GridSearchCV hyperparameter tuning.

## Data Source

Kaggle dataset at `assets/titanic.csv` ([source](https://www.kaggle.com/datasets/brendan45774/test-file)) — 891 passengers, 12 columns.

## Project Structure

```
titanic/
├── data/
│   └── data_preprocessed.csv   # Cleaned dataset (891 rows, 10 features + target)
├── experiments/
│   └── confusion_matrix.png    # Evaluation heatmap
├── features/                   # (planned) Feature engineering modules
├── models/                     # (planned) Model training scripts
├── preprocessing/              # (planned) Preprocessing utilities
├── trained_models/             # (planned) Serialized models
└── README.md
```

> The full pipeline currently lives in `backend/titanic.py` as a standalone script. The plan is to refactor it into the project modules above.

## Pipeline

The notebook `ml/jupyter/titanic.ipynb` and `backend/titanic.py` implement the same workflow:

### 1. Preprocessing

- **Dropped columns** (non-predictive): PassengerId, Name, Ticket, Cabin, Embarked
- **Missing ages**: filled with class-specific median (Class 1: 42, Class 2: 26.5, Class 3: 24)
- **Missing fare**: filled with overall median
- **Sex**: binary encoding (male=1, female=0)

### 2. Feature Engineering

| Feature | Description |
|---------|-------------|
| FamilySize | SibSp + Parch |
| IsAlone | 1 if FamilySize = 0 |
| FareBin | Quantile-based bucketing into 4 categories |
| AgeBin | Age groups: 0-12, 12-20, 20-40, 40-60, 60+ |

Final feature set (10 features): Pclass, Sex, Age, SibSp, Parch, Fare, FamilySize, IsAlone, FareBin, AgeBin.

### 3. Training

- Train/test split: 75/25 (`random_state=42`)
- MinMaxScaler normalization (fit on train, transform on both)
- KNeighborsClassifier with GridSearchCV (5-fold CV):
  - `n_neighbors`: 1–20
  - `metric`: euclidean, manhattan, minkowski
  - `weights`: uniform, distance

### 4. Evaluation

- Accuracy score
- Confusion matrix (seaborn heatmap saved to `experiments/confusion_matrix.png`)

## Running

```bash
# Standalone script
uv run python backend/titanic.py

# Or via Jupyter
uv sync && jupyter lab
# Then open ml/jupyter/titanic.ipynb
```

## Backend Integration

Not yet integrated into the FastAPI backend. The houses project shows the intended pattern: serialize the model, load at startup, serve via `/predict/titanic`.
