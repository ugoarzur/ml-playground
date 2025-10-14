import joblib
from sklearn.linear_model import LinearRegression


def train_model(x, y):
    model = LinearRegression()
    model.fit(x, y)
    joblib.dump(model, "ml/projects/houses/trained_models/model.pkl")
    return model
