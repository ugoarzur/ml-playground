import numpy as np
from fastapi import APIRouter

from backend.src.houses.schemas import HouseFeatures
from ml.projects.houses.data.loader import load_dataset
from ml.projects.houses.features.polynomial import create_polynomial_features
from ml.projects.houses.models.model_helper import load_model

router = APIRouter(prefix="/predict", tags=["houses"])

house_model = None
poly_transformer = None


def startup():
    """Load model and fit polynomial transformer. Called from app lifespan."""
    global house_model, poly_transformer
    house_model = load_model("hgbr_best.joblib")
    X, _ = load_dataset()
    _, poly_transformer = create_polynomial_features(X)


@router.post("/houses")
def predict_house(features: HouseFeatures):
    X = np.array(
        [
            [
                features.MedInc,
                features.HouseAge,
                features.AveRooms,
                features.AveBedrms,
                features.Population,
                features.AveOccup,
                features.Latitude,
                features.Longitude,
            ]
        ]
    )

    X_poly = poly_transformer.transform(X)
    prediction = house_model.predict(X_poly)
    price = prediction[0] * 100_000

    return {"predicted_price": round(price, 2)}
