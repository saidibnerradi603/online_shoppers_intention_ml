"""
Data Preprocessing Module

This module handles all data preprocessing steps based on the process developed in the 02_data_preprocessing.ipynb notebook.
"""
import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from utils.config import NUMERICAL_FEATURES,CATEGORICAL_FEATURES,BOOL_COL,TARGET_COL,ENCODERS_PATH,SCALER_PATH,PROCESSED_DATA_DIR,ONLINE_SHOPPERS_DATA



def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert encoded ID columns to categorical dtype
    
    Args:
        df: Raw dataframe
        
    Returns:
        DataFrame with converted data types
    """
    df_copy = df.copy()
    df_copy["OperatingSystems"] = df_copy["OperatingSystems"].astype('object')
    df_copy["Browser"] = df_copy["Browser"].astype('object')
    df_copy["Region"] = df_copy["Region"].astype('object')
    df_copy["TrafficType"] = df_copy["TrafficType"].astype('object')
    
    return df_copy


def handle_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from the dataset
    
    Args:
        df: Input dataframe
        
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates().reset_index(drop=True)


def encode_categorical_features(df: pd.DataFrame, save_encoders: bool = True, 
                               encoders_path: str = ENCODERS_PATH) -> Tuple[pd.DataFrame, Dict]:
    """
    Encode categorical variables using LabelEncoder
    
    Args:
        df: Input dataframe
        save_encoders: Whether to save the encoders
        encoders_path: Path to save the encoders
        
    Returns:
        Tuple of (processed_dataframe, encoders_dict)
    """
    df_processed = df.copy()
    encoders = {}
    
    # Convert Weekend and Revenue to numerical
    df_processed[BOOL_COL] = df_processed[BOOL_COL].map({'TRUE': 1, 'False': 0, True: 1, False: 0})
    df_processed[TARGET_COL] = df_processed[TARGET_COL].map({'TRUE': 1, 'False': 0, True: 1, False: 0})
    
    # Encode categorical features
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        encoders[col] = le
        
    # Save encoders if requested
    if save_encoders:
        os.makedirs(os.path.dirname(encoders_path), exist_ok=True)
        joblib.dump(encoders, encoders_path)
        
    return df_processed, encoders


def scale_features(df: pd.DataFrame, train: bool = True,
                  scaler_path: str = SCALER_PATH) -> pd.DataFrame:
    """
    Scale numerical features using StandardScaler
    
    Args:
        df: Input dataframe
        train: Whether this is training data (fit_transform) or test data (transform only)
        scaler_path: Path to save/load the scaler
        
    Returns:
        DataFrame with scaled features
    """
    df_scaled = df.copy()
    
    if train:
        scaler = StandardScaler()
        df_scaled[NUMERICAL_FEATURES] = scaler.fit_transform(df_scaled[NUMERICAL_FEATURES])
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
    else:
        scaler = joblib.load(scaler_path)
        df_scaled[NUMERICAL_FEATURES] = scaler.transform(df_scaled[NUMERICAL_FEATURES])
        
    return df_scaled


def split_data(X: pd.DataFrame, y: pd.Series, 
              test_size: float = 0.2, 
              random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and test sets with stratification
    
    Args:
        X: Features dataframe
        y: Target series
        test_size: Proportion of test data
        random_state: Random seed
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test


def preprocess_data(df: pd.DataFrame, 
                   train: bool = True,
                   encoders_path: str = ENCODERS_PATH,
                   scaler_path: str = SCALER_PATH) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Full preprocessing pipeline
    
    Args:
        df: Raw dataframe
        train: Whether this is training data
        encoders_path: Path to save/load encoders
        scaler_path: Path to save/load scaler
        
    Returns:
        Tuple of (X_processed, y_processed)
    """
    # Convert data types
    df = convert_data_types(df)
    
    # Handle duplicates
    df = handle_duplicates(df)
    
    # Encode categorical features
    if train:
        df, _ = encode_categorical_features(df, save_encoders=True, encoders_path=encoders_path)
    else:
        # Load encoders for inference
        encoders = joblib.load(encoders_path)
        df_processed = df.copy()
        
        # Convert Weekend to numerical
        df_processed[BOOL_COL] = df_processed[BOOL_COL].map({'TRUE': 1, 'False': 0, True: 1, False: 0})
        
        # Encode categorical features using loaded encoders
        for col, encoder in encoders.items():
            df_processed[col] = df_processed[col].apply(
                lambda x: encoder.transform([str(x)])[0] if str(x) in encoder.classes_ else -1
            )
        
        df = df_processed
    
    # Split features and target
    X = df.drop(columns=[TARGET_COL]) if TARGET_COL in df.columns else df
    y = df[TARGET_COL] if TARGET_COL in df.columns else None
    
    # Scale features
    X = scale_features(X, train=train, scaler_path=scaler_path)
    
    return X, y


def save_processed_data(X_train: pd.DataFrame, X_test: pd.DataFrame, 
                       y_train: pd.Series, y_test: pd.Series,
                       output_path: str = PROCESSED_DATA_DIR) -> None:
    """
    Save processed datasets to CSV files
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target
        output_path: Directory to save the files
    """
    os.makedirs(output_path, exist_ok=True)
    
    X_train.to_csv(os.path.join(output_path, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_path, 'X_test.csv'), index=False)
    pd.DataFrame(y_train).to_csv(os.path.join(output_path, 'y_train.csv'), index=False)
    pd.DataFrame(y_test).to_csv(os.path.join(output_path, 'y_test.csv'), index=False)
    
    print(f"Processed data saved to: {output_path}")


def run_preprocessing_pipeline(data_path: str = ONLINE_SHOPPERS_DATA,
                            output_path: str = PROCESSED_DATA_DIR,
                            test_size: float = 0.2,
                            random_state: int = 42) -> None:
    """
    Run the full preprocessing pipeline
    
    Args:
        data_path: Path to raw data
        output_path: Directory to save processed data
        test_size: Proportion of test data
        random_state: Random seed
    """
    # Load data
    df = pd.read_csv(data_path)
    print(f"Loaded data with shape: {df.shape}")
    
    # Preprocess
    df = convert_data_types(df)
    df = handle_duplicates(df)
    df, _ = encode_categorical_features(df)
    
    # Split features and target
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    
    # Split train/test
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size, random_state=random_state)
    
    # Scale features
    X_train = scale_features(X_train, train=True)
    X_test = scale_features(X_test, train=False)
    
    # Save processed data
    save_processed_data(X_train, X_test, y_train, y_test, output_path=output_path)
    
    print("Preprocessing pipeline completed successfully!")


if __name__ == "__main__":
    run_preprocessing_pipeline()
