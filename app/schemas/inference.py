from typing import List
from pydantic import BaseModel, Field, ConfigDict

class FlightPredictionRequest(BaseModel):
    """Input fields available before departure."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "airline": "AA",
            "origin_airport": "LAX",
            "destination_airport": "PBI",
            "month": 1,
            "day": 1,
            "day_of_week": 4,
            "scheduled_departure_hhmm": 10,
            "scheduled_time_minutes": 300,
            "distance": 2330,
            "cancelled": 0,
            "diverted": 0
        }
    })

    airline: str = Field(..., min_length=2, max_length=3, description="Airline code, e.g., AA, DL, WN")
    origin_airport: str = Field(..., min_length=3, max_length=5, description="Origin airport code, e.g., LAX")
    destination_airport: str = Field(..., min_length=3, max_length=5, description="Destination airport code, e.g., PBI")
    month: int = Field(..., ge=1, le=12)
    day: int = Field(..., ge=1, le=31)
    day_of_week: int = Field(..., ge=1, le=7, description="1=Monday, 7=Sunday")
    scheduled_departure_hhmm: int = Field(..., ge=0, le=2359, description="Scheduled departure as HHMM or hour")
    scheduled_time_minutes: float = Field(..., ge=0, description="Scheduled flight time in minutes")
    distance: float = Field(..., ge=0, description="Flight distance in miles")
    cancelled: int = Field(0, ge=0, le=1)
    diverted: int = Field(0, ge=0, le=1)

class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="1 = delayed >=15 minutes, 0 = on-time")
    label: str
    delay_probability: float
    threshold: float
    model_loaded: bool
    mode: str

class BatchPredictionRequest(BaseModel):
    flights: List[FlightPredictionRequest] = Field(..., min_length=1, max_length=100)

class BatchPredictionResponse(BaseModel):
    count: int
    predictions: List[PredictionResponse]
