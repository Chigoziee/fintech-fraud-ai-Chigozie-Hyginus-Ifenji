from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from utils.models import Txn_data, PredictionResponse
from utils.feature_engineering import feature_engineering as fe
from utils.predict import predict
from utils.explainer import explain_prediction
import logging


app = FastAPI(title="Fraud Detection API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


logging.basicConfig(level=logging.INFO)


@app.get("/")
def home():
    return {"message": "Fraud Detection API is live 🌍"}


@app.post("/predict", response_model=PredictionResponse)
def make_prediction(payload: Txn_data):
    try:
        logging.info("Received payload")

        modified_data = fe(payload)
        prediction = predict(modified_data)
        explanation, _ = explain_prediction(modified_data)
        explanation_cleaned = {k: float(v) for k, v in explanation.items()}

        return {
            "prediction": str(prediction),
            "explanation": explanation_cleaned
        }
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")