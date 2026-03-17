from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.src.houses.router import router as houses_router
from backend.src.houses.router import startup as houses_startup


@asynccontextmanager
async def lifespan(app: FastAPI):
    houses_startup()
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(houses_router)


@app.get("/")
def read_root():
    return {"Hello": "World"}
