import pathlib

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

from ml.shared.metrics import evaluate_model
from ml.shared.persistence import load_model as _load_model
from ml.shared.persistence import save_model as _save_model

TRAINED_MODELS_DIR = pathlib.Path(__file__).parent.parent / "trained_models"


def train_model(X, y, model=None):
    """Train a model on (X, y). Defaults to LinearRegression."""
    if model is None:
        model = LinearRegression()
    model.fit(X, y)
    return model


def compare_models(X_train, y_train, X_test, y_test):
    """Train and score LinearRegression, HGBR, and RandomForest. Return dict of scores."""
    results = {}
    models = [
        LinearRegression(),
        HistGradientBoostingRegressor(),
        RandomForestRegressor(n_jobs=-1),
    ]
    for m in models:
        m.fit(X_train, y_train)
        score = evaluate_model(m, X_test, y_test)
        results[str(m)] = score
    return results


def grid_search_hgbr(X_train, y_train, X_test, y_test, max_iters=None, learning_rates=None):
    """Manual grid search over HGBR hyperparameters.

    Returns (best_model, best_score, all_results).
    """
    if max_iters is None:
        max_iters = [250, 300, 350]
    if learning_rates is None:
        learning_rates = [0.1, 0.05, 0.001]

    best_score = -1.0
    best_model = None
    all_results = []

    for lr in learning_rates:
        for mi in max_iters:
            model = HistGradientBoostingRegressor(max_iter=mi, learning_rate=lr)
            model.fit(X_train, y_train)
            score = evaluate_model(model, X_test, y_test)
            all_results.append({"max_iter": mi, "learning_rate": lr, "r2_score": score})
            if score > best_score:
                best_score = score
                best_model = model

    return best_model, best_score, all_results


def save_model(model, filename="model.joblib"):
    """Save model to this project's trained_models/ directory."""
    return _save_model(model, TRAINED_MODELS_DIR / filename)


def load_model(filename="model.joblib"):
    """Load model from this project's trained_models/ directory."""
    return _load_model(TRAINED_MODELS_DIR / filename)
