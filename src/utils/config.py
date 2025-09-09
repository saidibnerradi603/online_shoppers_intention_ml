"""
Configuration and Path Utilities

This module provides consistent configuration and path management
across the application.
"""
import os
from pathlib import Path
from typing import Dict

# Root directory of the project
ROOT_DIR = Path(__file__).parent.parent.parent



# Data paths
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

ONLINE_SHOPPERS_DATA = RAW_DATA_DIR / "online_shoppers_intention.csv"

# Model paths
MODEL_DIR = ROOT_DIR / "models"
FINAL_MODEL_PATH = MODEL_DIR / "final_revenue_prediction_model.joblib"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
ENCODERS_PATH = MODEL_DIR / "label_encoders.pkl"

# Reports paths
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

# Ensure directories exist
def ensure_directories() -> None:
    """Create all required directories if they don't exist."""
    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODEL_DIR,
        FIGURES_DIR,
        METRICS_DIR
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        
    print("Directory structure verified.")

# Model configuration
MODEL_CONFIG = {
    "random_forest": {
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_split": 2,
    }
}

# Feature lists
NUMERICAL_FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay"
]





CATEGORICAL_FEATURES = [
    'Month',
    'OperatingSystems',
    'Browser',
    'Region',
    'TrafficType',
    'VisitorType',
    'Weekend'
]


BOOL_COL = 'Weekend'
TARGET_COL = 'Revenue'



# API Configuration
API_CONFIG = {
    "title": "Online Shoppers Revenue Prediction API",
    "description": "Predict whether an online shopper will make a purchase",
    "version": "1.0.0"
}

def get_project_paths() -> Dict[str, Path]:
    """
    Get a dictionary of project paths
    
    Returns:
        Dictionary mapping path names to Path objects
    """
    return {
        "root": ROOT_DIR,
        "data": DATA_DIR,
        "raw_data": RAW_DATA_DIR,
        "processed_data": PROCESSED_DATA_DIR,
        "models": MODEL_DIR,
        "reports": REPORTS_DIR,
        "figures": FIGURES_DIR,
        "metrics": METRICS_DIR,
        "final_model": FINAL_MODEL_PATH,
        "scaler": SCALER_PATH,
        "encoders": ENCODERS_PATH
    }



if __name__ == "__main__":
    ensure_directories()
    print("Project paths:")
    for name, path in get_project_paths().items():
        print(f"  {name}: {path}")
    
