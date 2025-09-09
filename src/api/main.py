"""
FastAPI Main Application

This module provides the FastAPI application and API endpoints
for the Online Shoppers Revenue Prediction model.
"""
from fastapi import FastAPI, HTTPException, Depends
from typing import Dict, Any
import pandas as pd
import joblib
import datetime
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from api.models import (
    HealthResponse, ModelInfoResponse, WelcomeResponse, 
    PredictionRequest, PredictionResponse, ErrorResponse
)
from utils.config import (
    API_CONFIG, NUMERICAL_FEATURES, CATEGORICAL_FEATURES,
    TARGET_COL, FINAL_MODEL_PATH, SCALER_PATH, ENCODERS_PATH
)
from prediction import load_model_artifacts, predict_revenue

app = FastAPI(
    title=API_CONFIG["title"],
    description=API_CONFIG["description"],
    version=API_CONFIG["version"],
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

model_artifacts = None

@app.on_event("startup")
async def startup_event():
    """Load model artifacts on startup"""
    global model_artifacts
    try:
        model_artifacts = load_model_artifacts(
            model_path=FINAL_MODEL_PATH,
            scaler_path=SCALER_PATH,
            encoders_path=ENCODERS_PATH
        )
        print("Model artifacts loaded successfully")
    except Exception as e:
        print(f"Error loading model artifacts: {str(e)}")
        model_artifacts = None


def get_artifacts():
    """
    Dependency to ensure model artifacts are loaded
    """
    if model_artifacts is None:
        raise HTTPException(status_code=500, detail="Model artifacts not loaded")
    return model_artifacts


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint to verify API is running
    """
    return {
        "status": "ok",
        "timestamp": datetime.datetime.now().isoformat()
    }


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info(artifacts: Dict[str, Any] = Depends(get_artifacts)):
    """
    Get model metadata and information
    """
    model = artifacts.get('model')
    
    return {
        "model_name": "Revenue Prediction Random Forest",
        "model_version": "1.0.0",
        "features": NUMERICAL_FEATURES + CATEGORICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
        "target": TARGET_COL,
        "creation_date": datetime.datetime.fromtimestamp(
            os.path.getmtime(FINAL_MODEL_PATH)
        ).strftime("%Y-%m-%d"),
        "description": "Random Forest model to predict whether a visitor will generate revenue"
    }


@app.get("/", response_model=WelcomeResponse, tags=["System"])
async def root():
    """
    Root endpoint with welcome message and API information
    """
    return {
        "title": API_CONFIG["title"],
        "description": API_CONFIG["description"],
        "version": API_CONFIG["version"],
        "endpoints": {
            "/": "Welcome message and API information",
            "/health": "Health check endpoint",
            "/model-info": "Model metadata and information",
            "/predict": "Make a prediction with the model",
            "/docs": "API documentation (Swagger UI)",
            "/redoc": "API documentation (ReDoc)"
        }
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(
    input_data: PredictionRequest, 
    artifacts: Dict[str, Any] = Depends(get_artifacts)
):
    """
    Make a prediction using the trained model
    """
    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data.dict()])
        
        # Make prediction
        prediction_result = predict_revenue(
            data=input_df,
            artifacts=artifacts,
            return_probabilities=True
        )
        
        # Extract prediction and probability
        prediction = prediction_result['predictions'][0]
        probability = prediction_result['probabilities'][0]
        
        # Return response
        return {
            "prediction": prediction,
            "probability": probability,
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {
        "error": "Request failed",
        "detail": exc.detail
    }


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return {
        "error": "Internal server error",
        "detail": str(exc)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
