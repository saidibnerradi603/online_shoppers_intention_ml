"""
Model Training Module

This module handles the model training based on the process developed 
in the 03_Model_Selection.ipynb and 04_Model Training and Interpretation.ipynb notebooks.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Any
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE

from utils.config  import PROCESSED_DATA_DIR,MODEL_CONFIG,FIGURES_DIR,FINAL_MODEL_PATH

def load_processed_data(processed_data_path: str = PROCESSED_DATA_DIR) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Load preprocessed data
    
    Args:
        processed_data_path: Path to processed data directory
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    X_train = pd.read_csv(os.path.join(processed_data_path, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(processed_data_path, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(processed_data_path, 'y_train.csv')).values.ravel()
    y_test = pd.read_csv(os.path.join(processed_data_path, 'y_test.csv')).values.ravel()
    
    return X_train, X_test, y_train, y_test


def apply_smote(X_train: pd.DataFrame, y_train: np.ndarray, random_state: int = 42) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Apply SMOTE to balance the training data
    
    Args:
        X_train: Training features
        y_train: Training target
        random_state: Random seed
        
    Returns:
        Tuple of (X_train_smote, y_train_smote)
    """
    smote = SMOTE(random_state=random_state, sampling_strategy="auto")
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    return X_train_smote, y_train_smote


def train_random_forest_model(X_train: pd.DataFrame, y_train: np.ndarray, 
                             n_estimators: int = MODEL_CONFIG["random_forest"]["n_estimators"], 
                             max_depth: int = MODEL_CONFIG["random_forest"]["max_depth"], 
                             min_samples_split: int = MODEL_CONFIG["random_forest"]["min_samples_split"], 
                             random_state: int = 42) -> RandomForestClassifier:
    """
    Train the Random Forest model with the optimal parameters
    
    Args:
        X_train: Training features
        y_train: Training target
        n_estimators: Number of trees
        max_depth: Maximum depth of trees
        min_samples_split: Minimum samples required to split
        random_state: Random seed
        
    Returns:
        Trained Random Forest model
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: RandomForestClassifier, X_test: pd.DataFrame, y_test: np.ndarray, 
                  output_path: str = FIGURES_DIR) -> Dict[str, Any]:
    """
    Evaluate the model on test data and generate visualizations
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        output_path: Directory to save visualizations
        
    Returns:
        Dictionary with evaluation metrics
    """
    os.makedirs(output_path, exist_ok=True)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Classification report
    class_report = classification_report(y_test, y_pred, target_names=['No Revenue', 'Revenue'], output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Revenue', 'Revenue'],
                yticklabels=['No Revenue', 'Revenue'])
    plt.title('Final Model Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(output_path, "final_model_confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(output_path, "final_model_roc_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Feature importance
    importances = model.feature_importances_
    feature_names = X_test.columns
    
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance_df, palette='viridis')
    plt.title('Feature Importance for Revenue Prediction', fontsize=16)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.savefig(os.path.join(output_path, "Feature_importance_for_Revenue_Prediction.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Return evaluation results
    evaluation_results = {
        'classification_report': class_report,
        'confusion_matrix': cm.tolist(),
        'roc_auc': roc_auc,
        'feature_importance': feature_importance_df.to_dict('records')
    }
    
    return evaluation_results


def save_model(model: RandomForestClassifier, model_path: str = FINAL_MODEL_PATH) -> None:
    """
    Save the trained model
    
    Args:
        model: Trained model
        model_path: Path to save the model
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model successfully saved to: {model_path}")


def train_and_evaluate_model(processed_data_path: str = PROCESSED_DATA_DIR,
                           use_smote: bool = True,
                           model_path: str = FINAL_MODEL_PATH,
                           figures_path: str = FIGURES_DIR) -> Dict[str, Any]:
    """
    Run the full model training and evaluation pipeline
    
    Args:
        processed_data_path: Path to processed data
        use_smote: Whether to use SMOTE for balancing
        model_path: Path to save the model
        figures_path: Path to save evaluation figures
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Load data
    X_train, X_test, y_train, y_test = load_processed_data(processed_data_path)
    print(f"Loaded processed data: X_train={X_train.shape}, X_test={X_test.shape}")
    
    # Apply SMOTE if requested
    if use_smote:
        X_train, y_train = apply_smote(X_train, y_train)
        print(f"Applied SMOTE: X_train={X_train.shape}")
    
    # Train model
    print("Training Random Forest model...")
    model = train_random_forest_model(X_train, y_train)
    
    # Evaluate model
    print("Evaluating model...")
    evaluation_metrics = evaluate_model(model, X_test, y_test, figures_path)
    
    # Save model
    save_model(model, model_path)
    
    return evaluation_metrics


if __name__ == "__main__":
    metrics = train_and_evaluate_model()
    print(f"Model F1 Score: {metrics['classification_report']['Revenue']['f1-score']:.4f}")
    print(f"Model ROC AUC: {metrics['roc_auc']:.4f}")
