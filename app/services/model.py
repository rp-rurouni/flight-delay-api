from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd

from app.config import MODEL_PATH


class FlightDelayModelService:
    """Loads the trained flight delay bundle and runs inference.

    Expected bundle keys from the Colab notebook:
    model, threshold, feature_cols, origin_delay_map, dest_delay_map,
    origin_fallback, dest_fallback.
    """

    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = Path(model_path)
        self.model = None
        self.threshold = 0.32077
        self.feature_cols: List[str] = []
        self.origin_delay_map: Dict[str, float] = {}
        self.dest_delay_map: Dict[str, float] = {}
        self.origin_fallback = 0.18
        self.dest_fallback = 0.18
        self._initialized = False
        self.mode = "not_initialized"

    def initialize(self) -> None:
        """Load the trained joblib bundle once. Falls back to demo mode if missing."""
        if self._initialized:
            return

        if self.model_path.exists():
            bundle = joblib.load(self.model_path)
            self.model = bundle["model"]
            self.threshold = float(bundle.get("threshold", self.threshold))
            self.feature_cols = list(bundle["feature_cols"])
            self.origin_delay_map = bundle.get("origin_delay_map", {})
            self.dest_delay_map = bundle.get("dest_delay_map", {})
            self.origin_fallback = float(bundle.get("origin_fallback", self.origin_fallback))
            self.dest_fallback = float(bundle.get("dest_fallback", self.dest_fallback))
            self.mode = "trained_model"
        else:
            # Keeps /health and /docs usable before the student copies the real model file.
            # README explains that the real submission should include the Colab-exported joblib file.
            self.mode = "demo_fallback_model_missing"

        self._initialized = True

    @property
    def model_loaded(self) -> bool:
        self.initialize()
        return self.model is not None

    def _departure_hour(self, hhmm: int) -> int:
        if 0 <= hhmm <= 23:
            hour = hhmm
        else:
            hour = hhmm // 100
        return min(max(int(hour), 0), 23)

    def _preprocess(self, raw: Dict[str, Any]) -> pd.DataFrame:
        """Match the notebook's inference preprocessing."""
        self.initialize()

        if not self.feature_cols:
            # Demo mode feature list, only used when the real bundle is absent.
            self.feature_cols = [
                "MONTH", "DAY", "DAY_OF_WEEK", "SCHEDULED_TIME", "DISTANCE",
                "CANCELLED", "DIVERTED", "DEP_HOUR", "IS_WEEKEND",
                "ORIGIN_ENCODED", "DEST_ENCODED"
            ]

        dep_hour = self._departure_hour(int(raw["scheduled_departure_hhmm"]))
        is_weekend = 1 if int(raw["day_of_week"]) in [6, 7] else 0
        origin = str(raw["origin_airport"]).upper()
        dest = str(raw["destination_airport"]).upper()
        airline = str(raw["airline"]).upper()

        row = {c: 0 for c in self.feature_cols}
        base_values = {
            "MONTH": raw["month"],
            "DAY": raw["day"],
            "DAY_OF_WEEK": raw["day_of_week"],
            "SCHEDULED_TIME": raw["scheduled_time_minutes"],
            "DISTANCE": raw["distance"],
            "CANCELLED": raw.get("cancelled", 0),
            "DIVERTED": raw.get("diverted", 0),
            "DEP_HOUR": dep_hour,
            "IS_WEEKEND": is_weekend,
            "ORIGIN_ENCODED": self.origin_delay_map.get(origin, self.origin_fallback),
            "DEST_ENCODED": self.dest_delay_map.get(dest, self.dest_fallback),
        }
        for key, value in base_values.items():
            if key in row:
                row[key] = value

        airline_col = f"AIRLINE_{airline}"
        if airline_col in row:
            row[airline_col] = 1

        return pd.DataFrame([row], columns=self.feature_cols)

    def _demo_probability(self, raw: Dict[str, Any]) -> float:
        """Simple fallback only for local API testing before adding real model."""
        prob = 0.18
        hour = self._departure_hour(int(raw["scheduled_departure_hhmm"]))
        if hour >= 16:
            prob += 0.08
        if raw["month"] in [6, 7, 8, 12]:
            prob += 0.04
        if raw["distance"] > 1500:
            prob += 0.03
        if raw["day_of_week"] in [5, 7]:
            prob += 0.02
        return max(0.01, min(prob, 0.95))

    def predict(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        self.initialize()
        raw = dict(raw)
        raw["airline"] = raw["airline"].upper()
        raw["origin_airport"] = raw["origin_airport"].upper()
        raw["destination_airport"] = raw["destination_airport"].upper()

        if self.model is not None:
            X = self._preprocess(raw)
            probability = float(self.model.predict_proba(X)[0, 1])
        else:
            probability = self._demo_probability(raw)

        prediction = int(probability >= self.threshold)
        label = "Delayed (>=15 min arrival delay)" if prediction == 1 else "On-time (<15 min arrival delay)"
        return {
            "prediction": prediction,
            "label": label,
            "delay_probability": round(probability, 6),
            "threshold": self.threshold,
            "model_loaded": self.model is not None,
            "mode": self.mode,
        }


_model_service: FlightDelayModelService | None = None


def get_model_service() -> FlightDelayModelService:
    global _model_service
    if _model_service is None:
        _model_service = FlightDelayModelService()
    return _model_service
