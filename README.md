# Online Shoppers Revenue Prediction

A machine learning project to predict whether an online shopper will generate revenue (make a purchase) based on session data.

## Project Overview

This project develops a machine learning model to predict whether an online shopper will generate revenue (make a purchase) based on their browsing behavior and session information. Using the "Online Shoppers Intention" dataset, a Random Forest classifier is trained to identify patterns that lead to purchases, which can help e-commerce businesses optimize their websites and marketing strategies to increase conversion rates.

## Table of Contents

- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Training](#model-training)
  - [Making Predictions](#making-predictions)
  - [API Server](#api-server)
- [Model Performance](#model-performance)
- [API Documentation](#api-documentation)
- [Development Process](#development-process)

## Project Structure

```
online_shoppers_intention_ml/
├── data/
│   ├── processed/            # Processed datasets ready for modeling
│   │   ├── X_test.csv
│   │   ├── X_train.csv
│   │   ├── y_test.csv
│   │   └── y_train.csv
│   └── raw/                  # Raw dataset
│       └── online_shoppers_intention.csv
├── models/                   # Saved model artifacts
│   ├── final_revenue_prediction_model.joblib
│   ├── label_encoders.pkl
│   └── scaler.pkl
├── notebooks/                # Jupyter notebooks for exploration and development
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_Model_Selection.ipynb
│   ├── 04_Model Training and Interpretation.ipynb
│   └── 05_Model Deployment.ipynb
├── reports/                  # Visualizations and metrics
│   ├── figures/
│   │   ├── categorical_distributions.png
│   │   ├── Feature_importance_for_Revenue_Prediction.png
│   │   ├── final_model_confusion_matrix.png
│   │   ├── final_model_roc_curve.png
│   │   ├── Pairwise_Correlation_Matrix.png
│   │   ├── scatter_plots.png
│   │   ├── target_distribution.png
│   │   └── test_f1_scores_comparison.png
│   └── metrics/
│       └── results.csv
├── src/                      # Source code
│   ├── api/                  # API implementation
│   │   ├── __init__.py
│   │   ├── main.py          # FastAPI application
│   │   ├── models.py        # Pydantic models for API
│   │   └── server.py        # Uvicorn server
│   ├── utils/               # Utility modules
│   │   ├── __init__.py
│   │   └── config.py        # Configuration and paths
│   ├── preprocessing.py     # Data preprocessing pipeline
│   ├── training.py         # Model training pipeline
│   └── prediction.py       # Prediction functionality
└── README.md                # Project documentation
```

## Dataset

The project uses the "Online Shoppers Intention" dataset, which contains information about user sessions on an e-commerce website. Each session is labeled with whether it resulted in a purchase (Revenue = TRUE) or not (Revenue = FALSE).

Dataset Features:
- **Administrative**: Number of pages in administrative category visited
- **Administrative_Duration**: Time spent on administrative pages
- **Informational**: Number of pages in informational category visited
- **Informational_Duration**: Time spent on informational pages
- **ProductRelated**: Number of product-related pages visited
- **ProductRelated_Duration**: Time spent on product-related pages
- **BounceRates**: Percentage of visitors who enter the site from a page and then leave without triggering any other requests
- **ExitRates**: Percentage of pageviews that were the last in the session
- **PageValues**: Average value for a web page that a user visited before completing an e-commerce transaction
- **SpecialDay**: Closeness of site visiting time to a special day (e.g., Mother's Day, Valentine's Day)
- **Month**: Month of the year
- **OperatingSystems**: Operating system used by the visitor
- **Browser**: Browser used by the visitor
- **Region**: Geographic region
- **TrafficType**: Traffic source category
- **VisitorType**: Type of visitor (New, Returning, Other)
- **Weekend**: Whether the session occurred on a weekend
- **Revenue**: Whether the session resulted in a purchase (Target variable)

## Features

The project includes the following key features:

1. **Data Preprocessing Pipeline**:
   - Type conversion for numerical and categorical features
   - Duplicate removal
   - Categorical feature encoding
   - Feature scaling
   - Train/test splitting with stratification

2. **Machine Learning Model**:
   - Random Forest classifier
   - SMOTE for handling class imbalance
   - Feature importance analysis

3. **Model Evaluation**:
   - Classification metrics (precision, recall, F1-score)
   - Confusion matrix
   - ROC curve and AUC score

4. **API Integration**:
   - RESTful API with FastAPI
   - Input validation with Pydantic models
   - Endpoints for health check, model info, and predictions
   - Swagger documentation

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd online_shoppers_intention_ml
```

2. Create and activate a virtual environment:
```bash
# Using venv
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix/MacOS
source venv/bin/activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preprocessing

To preprocess the raw data and generate train/test splits:

```bash
python src/preprocessing.py
```

This will:
- Load the raw dataset
- Convert data types
- Remove duplicates
- Encode categorical features
- Scale numerical features
- Split into train/test sets
- Save processed data to the `data/processed` directory

### Model Training

To train the Random Forest model:

```bash
python src/training.py
```

This will:
- Load preprocessed data
- Apply SMOTE to balance the training data
- Train a Random Forest classifier
- Evaluate the model on the test set
- Generate evaluation visualizations
- Save the trained model to `models/final_revenue_prediction_model.joblib`


### API Server

To start the FastAPI server:

```bash
python -m src.api.server
```

The API will be available at `http://localhost:8000` with the following endpoints:
- `/`: Welcome message and API information
- `/health`: Health check endpoint
- `/model-info`: Model metadata and information
- `/predict`: Make a prediction with the model
- `/docs`: Interactive API documentation (Swagger UI)
- `/redoc`: Alternative API documentation (ReDoc)

## Model Performance

Based on the model evaluation results, the Random Forest classifier with SMOTE balancing achieved the best performance:

| Model | Balancing Method | Test F1-Score | CV Best F1-Score |
|-------|-----------------|---------------|-----------------|
| Random Forest | SMOTE | 0.6667 | 0.9310 |
| Random Forest | ADASYN | 0.6705 | 0.9324 |
| Random Forest | Random Oversampler | 0.6640 | 0.9646 |
| Random Forest | None (Imbalanced) | 0.6676 | 0.6514 |

Key findings from model evaluation:
- The model achieved an F1 score of approximately 0.67 on the test set
- The most important features for prediction are:
  - PageValues
  - ProductRelated_Duration
  - BounceRates
  - ExitRates
- The model has good balance between precision and recall

## API Documentation

The API provides the following endpoints:

1. **GET `/`**:
   - Welcome message with API information and available endpoints

2. **GET `/health`**:
   - Health check endpoint
   - Returns status and timestamp

3. **GET `/model-info`**:
   - Information about the trained model
   - Returns model name, version, features, and metadata

4. **POST `/predict`**:
   - Make a prediction with the model
   - Request body must contain all required features
   - Returns prediction (Revenue/No Revenue) and probability

Interactive API documentation is available at `/docs` when the server is running.

## Development Process

The project was developed through the following steps:

1. **Data Exploration**: 
   - Analysis of feature distributions and relationships
   - Identification of patterns and correlations
   - Target variable distribution analysis

2. **Data Preprocessing**:
   - Feature engineering and transformation
   - Encoding categorical variables
   - Handling special cases

3. **Model Selection**:
   - Testing multiple algorithms (Logistic Regression, Decision Trees, Random Forest)
   - Hyperparameter tuning with cross-validation
   - Performance comparison across different models and balancing techniques

4. **Model Training and Interpretation**:
   - Training the final model with optimized parameters
   - Feature importance analysis
   - Performance visualization

5. **Model Deployment**:
   - Creating prediction pipeline
   - Developing FastAPI endpoints
   - Setting up the server

---

