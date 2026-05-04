from fastapi import APIRouter, HTTPException

from app.schemas.inference import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    FlightPredictionRequest,
    PredictionResponse,
)
from app.services.model import get_model_service

router = APIRouter()

@router.get("/health")
async def health_check():
    """Check that the API is running and report model status."""
    service = get_model_service()
    service.initialize()
    return {
        "status": "healthy",
        "model": "flight-delay-lightgbm",
        "model_loaded": service.model_loaded,
        "mode": service.mode,
    }

@router.post("/predict", response_model=PredictionResponse)
async def predict(request: FlightPredictionRequest):
    """Predict whether one flight will arrive 15+ minutes late."""
    try:
        service = get_model_service()
        return service.predict(request.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Bonus endpoint: predict delay risk for multiple flights."""
    try:
        service = get_model_service()
        predictions = [service.predict(f.model_dump()) for f in request.flights]
        return {"count": len(predictions), "predictions": predictions}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {exc}") from exc
