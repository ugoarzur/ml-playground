from typing import Union

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_pwid: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}



from fastapi import APIRouter
import joblib

router = APIRouter()
model = joblib.load('ml/projects/houses/trained_models/model.pkl')

@router.post("/predict/houses")
def predict(features: HouseFeatures):
    prediction = model.predict([features.dict().values()])
    return {"predicted_price": prediction[0]}
