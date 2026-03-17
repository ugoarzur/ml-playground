# Houses — California Housing Price Prediction

Predicts median house prices in California using the `sklearn.datasets.fetch_california_housing` dataset (20,640 samples, 8 features).

## Project Structure

```
houses/
├── data/loader.py              # Loads the California Housing dataset
├── features/polynomial.py      # Polynomial feature generation (degree 2: 8 → 45 features)
├── models/model_helper.py      # Training, model comparison, and grid search
├── preprocessing/              # (planned) Dedicated preprocessing steps
├── trained_models/             # Serialized models (.joblib)
│   └── hgbr_best.joblib        # Best model (HistGradientBoostingRegressor, R² ≈ 0.844)
└── README.md
```

## Training Pipeline

The full pipeline is executed in `ml/jupyter/house_pricing_guess.ipynb`:

1. **Load** — `data/loader.py` fetches the dataset via scikit-learn
2. **Split** — 80/20 train/test (`random_state=432`)
3. **Feature engineering** — `features/polynomial.py` applies `PolynomialFeatures(degree=2)` to capture interactions between variables
4. **Model comparison** — `models/model_helper.py` trains and evaluates:
   - `LinearRegression` → R² ≈ 0.661
   - `HistGradientBoostingRegressor` → R² ≈ 0.837
   - `RandomForestRegressor` → R² ≈ 0.804
5. **Grid search** — Hyperparameter optimization on HGBR (`max_iter`, `learning_rate`) → **R² ≈ 0.844**
6. **Export** — Saves the best model to `trained_models/hgbr_best.joblib`

## Running the Notebook

```bash
uv sync && jupyter lab
```

Then open `ml/jupyter/house_pricing_guess.ipynb`.

## Serving Predictions via the API

```bash
uv run fastapi dev backend/main.py
```

The `POST /predict/houses` endpoint expects a JSON body with the 8 features:

```json
{
  "MedInc": 3.5,
  "HouseAge": 30.0,
  "AveRooms": 5.0,
  "AveBedrms": 1.0,
  "Population": 1200.0,
  "AveOccup": 3.0,
  "Latitude": 37.5,
  "Longitude": -122.0
}
```

The server loads `hgbr_best.joblib` at startup, applies the same polynomial transformation used during training, and returns the predicted price in dollars.
