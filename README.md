# Flight Delay Prediction API

This project serves my AIM240 flight delay prediction model as a REST API for the AIM230 ML API Service assignment.

The model predicts whether a U.S. domestic flight is likely to arrive **15 minutes or more late** using pre-departure inputs such as airline, origin, destination, scheduled departure time, scheduled flight time, and distance.

## Coverage per Rubric

- Clean project structure: `app/`, `schemas/`, `services/`, `api/`, `models/`
- ML model service: loads `models/flight_delay_bundle.joblib`
- API endpoints: health check, single prediction, and batch prediction
- Input validation: Pydantic request/response schemas
- Docker: `Dockerfile` and `docker-compose.yml`
- Documentation: this README with run commands and examples
- Bonus attempt: batch prediction endpoint and basic tests

## Project Structure

```text
flight-delay-api/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── api/endpoints.py
│   ├── schemas/inference.py
│   └── services/model.py
├── models/
│   └── flight_delay_bundle.joblib   # added from Colab
├── notebooks/export_model_from_colab.py, RAP-Finalv1.1-Predicting Flight Delays Using Machine Learning.ipynb
├── tests/test_api.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Add the Trained Model

In the Colab notebook, run the code in:

```text
notebooks/export_model_from_colab.py
```

This downloads:

```text
flight_delay_bundle.joblib
```

Copy that file into:

```text
models/flight_delay_bundle.joblib
```

The API has a demo fallback if the model is missing, but the final submission includes the real `flight_delay_bundle.joblib` file.

## Run Locally

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open Swagger docs:

```text
http://localhost:8000/docs
```

Health check:

```text
http://localhost:8000/api/v1/health
```

## Run with Docker

```bash
docker compose up --build
```

Then open:

```text
http://localhost:8000/docs
```

## Example Prediction Request

Endpoint:

```text
POST /api/v1/predict
```

JSON body:

```json
{
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
```

Example response:

```json
{
  "prediction": 0,
  "label": "On-time (<15 min arrival delay)",
  "delay_probability": 0.2312,
  "threshold": 0.32077,
  "model_loaded": true,
  "mode": "trained_model"
}
```

## Example curl Command

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

## Batch Prediction Endpoint

Endpoint:

```text
POST /api/v1/predict/batch
```

Body:

```json
{
  "flights": [
    {
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
  ]
}
```

## Screenshots and for Submission

1. GitHub repository link
2. Screenshot of Swagger docs at `http://localhost:8000/docs`
3. Screenshot of successful `/api/v1/health`
4. Screenshot of successful `/api/v1/predict`


## Notes

This API intentionally uses only pre-departure fields to avoid target leakage. The model is loaded lazily and also initialized on startup. The prediction output includes the probability and threshold to make the result easier to explain.
