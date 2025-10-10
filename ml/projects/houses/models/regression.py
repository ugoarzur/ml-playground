import joblib
from sklearn.linear_model import LinearRegression


def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, 'ml/projects/houses/trained_models/model.pkl')
    return model
