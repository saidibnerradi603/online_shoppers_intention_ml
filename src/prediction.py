"""
Prediction Service

This module handles making predictions using the trained model,
"""
import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, List, Union, Any
from utils.config import FINAL_MODEL_PATH,SCALER_PATH,ENCODERS_PATH,BOOL_COL,NUMERICAL_FEATURES

def load_model_artifacts(
    model_path: str = FINAL_MODEL_PATH,
    scaler_path: str =SCALER_PATH,
    encoders_path: str = ENCODERS_PATH
) -> Dict[str, Any]:
    """
    Load all required artifacts for prediction
    
    Args:
        model_path: Path to the saved model
        scaler_path: Path to the saved scaler
        encoders_path: Path to the saved encoders
        
    Returns:
        Dictionary containing model, scaler and encoders
    """
    artifacts = {
        'model': joblib.load(model_path),
        'scaler': joblib.load(scaler_path),
        'encoders': joblib.load(encoders_path)
    }
    
    return artifacts


def preprocess_for_prediction(
    data: pd.DataFrame, 
    scaler: Any,
    encoders: Dict[str, Any]
) -> pd.DataFrame:
    """
    Preprocess new data for prediction
    
    Args:
        data: Raw input data
        scaler: Trained StandardScaler
        encoders: Dictionary of trained LabelEncoders
        
    Returns:
        Preprocessed data ready for prediction
    """
    processed_data = data.copy()
    
    # --- Step 1: Encode Categorical Variables ---
    for col, encoder in encoders.items():
        processed_data[col] = processed_data[col].apply(
            lambda x: encoder.transform([str(x)])[0] if str(x) in encoder.classes_ else -1
        )
    
    # --- Step 2: Encode Boolean Variable ---
    processed_data[BOOL_COL] = processed_data[BOOL_COL].map(
        {'TRUE': 1, 'False': 0, True: 1, False: 0}
    )
    
    # --- Step 3: Scale Numerical Features ---
  
    processed_data[NUMERICAL_FEATURES] = scaler.transform(processed_data[NUMERICAL_FEATURES])
    
    # --- Step 4: Ensure Column Order ---
    training_columns = [
        "Administrative", "Administrative_Duration", "Informational",
        "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
        "BounceRates", "ExitRates", "PageValues", "SpecialDay", "Month",
        "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType", "Weekend"
    ]
    
    processed_data = processed_data[training_columns]
    
    return processed_data


def predict_revenue(
    data: pd.DataFrame,
    artifacts: Dict[str, Any] = None,
    return_probabilities: bool = False
) -> Union[List[str], Dict[str, List]]:
    """
    Make revenue predictions on new data
    
    Args:
        data: Raw input data
        artifacts: Dictionary of model artifacts (if None, will load from default paths)
        return_probabilities: Whether to return prediction probabilities
        
    Returns:
        List of predictions or dictionary with predictions and probabilities
    """
    # Load artifacts if not provided
    if artifacts is None:
        artifacts = load_model_artifacts()
    
    model = artifacts['model']
    scaler = artifacts['scaler']
    encoders = artifacts['encoders']
    
    # Preprocess data
    processed_data = preprocess_for_prediction(data, scaler, encoders)
    
    # Make predictions
    predictions_numeric = model.predict(processed_data)
    
    # Convert to human-readable format
    predictions = ['Revenue' if pred == 1 else 'No Revenue' for pred in predictions_numeric]
    
    if return_probabilities:
        # Get probability of positive class (Revenue)
        probabilities = model.predict_proba(processed_data)[:, 1].tolist()
        return {
            'predictions': predictions,
            'probabilities': probabilities
        }
    
    return predictions


if __name__ == "__main__":
    """
    Example usage of the prediction module
    """
    # Create sample data for prediction (one visitor session)
    sample_data = pd.DataFrame({
        'Administrative': [2], 
        'Administrative_Duration': [80.0],
        'Informational': [0], 
        'Informational_Duration': [0.0],
        'ProductRelated': [10], 
        'ProductRelated_Duration': [600.0],
        'BounceRates': [0.2], 
        'ExitRates': [0.2],
        'PageValues': [8.0], 
        'SpecialDay': [0.0],
        'Month': ['May'], 
        'OperatingSystems': [2],
        'Browser': [1], 
        'Region': [1],
        'TrafficType': [2], 
        'VisitorType': ['Returning_Visitor'],
        'Weekend': [True]
    })
    
    print("Sample visitor session data:")
    print(sample_data)
    
    # Load model artifacts once
    artifacts = load_model_artifacts()
    
    # Make prediction
    result = predict_revenue(sample_data, artifacts, return_probabilities=True)
    
    # Display results
    print("\nPrediction Results:")
    print(f"Prediction: {result['predictions'][0]}")
    print(f"Probability of Revenue: {result['probabilities'][0]:.4f}")
    
    