import time
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response


app = FastAPI(
    title="Fraud Detection Inference API",
    description="MLflow-based fraud detection API with Prometheus monitoring",
    version="1.0.0"
)

model = joblib.load("sourceCode/modelFiles/lightgbmBalancedModel.pkl")
featureColumns = joblib.load("sourceCode/modelFiles/featureColumns.pkl")

requestCounter = Counter(
    "fraud_api_request_total",
    "Total number of API requests"
)

errorCounter = Counter(
    "fraud_api_error_total",
    "Total number of API errors"
)

predictionCounter = Counter(
    "fraud_prediction_total",
    "Total predictions by class",
    ["predictionClass"]
)

latencyHistogram = Histogram(
    "fraud_api_latency_seconds",
    "API latency in seconds"
)


@app.get("/")
def healthCheck():
    return {
        "status": "running",
        "message": "Fraud Detection API is active"
    }


@app.post("/predict")
def predictFraud(inputData: dict):
    startTime = time.time()
    requestCounter.inc()

    try:
        inputDataFrame = pd.DataFrame([inputData])

        for columnName in featureColumns:
            if columnName not in inputDataFrame.columns:
                inputDataFrame[columnName] = 0

        inputDataFrame = inputDataFrame[featureColumns]

        prediction = model.predict(inputDataFrame)[0]
        probability = model.predict_proba(inputDataFrame)[0][1]

        predictionLabel = "fraud" if prediction == 1 else "notFraud"

        predictionCounter.labels(predictionClass=predictionLabel).inc()

        latencyHistogram.observe(time.time() - startTime)

        return {
            "prediction": int(prediction),
            "predictionLabel": predictionLabel,
            "fraudProbability": float(probability)
        }

    except Exception as error:
        errorCounter.inc()
        latencyHistogram.observe(time.time() - startTime)
        raise HTTPException(status_code=500, detail=str(error))


@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
