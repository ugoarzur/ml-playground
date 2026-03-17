from sklearn.metrics import r2_score


def evaluate_model(model, X_test, y_test, metric=r2_score):
    """Score model predictions against y_test using the given metric (default: r2_score)."""
    y_pred = model.predict(X_test)
    return metric(y_test, y_pred)
