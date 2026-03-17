from pydantic import BaseModel, Field


class HouseFeatures(BaseModel):
    MedInc: float = Field(examples=[8.3252])
    HouseAge: float = Field(examples=[41.0])
    AveRooms: float = Field(examples=[6.98])
    AveBedrms: float = Field(examples=[1.02])
    Population: float = Field(examples=[322.0])
    AveOccup: float = Field(examples=[2.55])
    Latitude: float = Field(examples=[37.88])
    Longitude: float = Field(examples=[-122.23])
