from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import settings
from .feedback import append_feedback
from .inference import FraudPredictor

app = FastAPI(title="Fraud Detection API", version="0.1.0")


class Transaction(BaseModel):
    Time: float = Field(ge=0)
    Amount: float = Field(ge=0)
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float


class BatchRequest(BaseModel):
    transactions: list[Transaction] = Field(min_length=1, max_length=1000)


class FeedbackRequest(BaseModel):
    transaction: Transaction
    confirmed_fraud: bool


class PredictionResponse(BaseModel):
    fraud: bool
    score: float
    threshold: float
    model: str


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]


@lru_cache
def predictor() -> FraudPredictor:
    return FraudPredictor()


@app.get("/health")
def health():
    return {"status": "ok", "model_ready": settings.artifact_path.exists()}


@app.get("/model/info")
def model_info():
    try:
        artifact = predictor().artifact
        return {key: artifact[key] for key in ("version", "created_at", "model_name", "threshold", "feature_count", "dataset")}
    except FileNotFoundError as exc:
        raise HTTPException(503, "Modelo ainda não treinado.") from exc


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    try:
        return predictor().predict([transaction.model_dump()])[0]
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(503, str(exc)) from exc


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest):
    try:
        return {"predictions": predictor().predict([item.model_dump() for item in request.transactions])}
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(503, str(exc)) from exc


@app.post("/feedback", status_code=202)
def feedback(request: FeedbackRequest):
    append_feedback(request.transaction.model_dump(), int(request.confirmed_fraud))
    return {"accepted": True, "message": "Feedback registrado para o próximo ciclo incremental."}
