from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.endpoints import router
from app.config import APP_NAME, APP_VERSION
from app.services.model import get_model_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    service = get_model_service()
    service.initialize()
    print(f"Model service initialized in mode: {service.mode}")
    yield

app = FastAPI(
    title=APP_NAME,
    description="REST API that exposes a flight delay prediction model trained in the AIM240 capstone notebook.",
    version=APP_VERSION,
    lifespan=lifespan,
)

app.include_router(router, prefix="/api/v1", tags=["inference"])

@app.get("/")
async def root():
    return {"message": "Flight Delay Prediction API", "docs": "/docs", "health": "/api/v1/health"}
