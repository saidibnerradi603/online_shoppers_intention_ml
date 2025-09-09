"""
Pydantic Models for API Validation

This module defines the Pydantic models used for request validation
and response formatting in the FastAPI endpoints.
"""
from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any, List, Optional, Union
import datetime

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    timestamp: str

class ModelInfoResponse(BaseModel):
    """Model information response model"""
    model_name: str
    model_version: str
    features: List[str]
    categorical_features: List[str]
    numerical_features: List[str]
    target: str
    creation_date: str
    description: str

class WelcomeResponse(BaseModel):
    """Welcome message response model"""
    title: str
    description: str
    version: str
    endpoints: Dict[str, str]

class PredictionRequest(BaseModel):
    """
    Prediction request model with all required features
    """
    Administrative: int = Field(2, description="Number of administrative pages visited")
    Administrative_Duration: float = Field(80.0, description="Total time spent on administrative pages")
    Informational: int = Field(0, description="Number of informational pages visited")
    Informational_Duration: float = Field(0.0, description="Total time spent on informational pages")
    ProductRelated: int = Field(10, description="Number of product related pages visited")
    ProductRelated_Duration: float = Field(600.0, description="Total time spent on product related pages")
    BounceRates: float = Field(0.2, description="Bounce rate of the visitor")
    ExitRates: float = Field(0.2, description="Exit rate of the visitor")
    PageValues: float = Field(8.0, description="Page value of the visitor")
    SpecialDay: float = Field(0.0, ge=0, le=1, description="Closeness to a special day (0-1)")
    Month: str = Field("May", description="Month of the visit")
    OperatingSystems: int = Field(2, description="Operating system of the visitor")
    Browser: int = Field(1, description="Browser of the visitor")
    Region: int = Field(1, description="Region of the visitor")
    TrafficType: int = Field(2, description="Traffic type")
    VisitorType: str = Field("Returning_Visitor", description="Type of visitor")
    Weekend: bool = Field(True, description="Whether the visit was on a weekend")
    
    @field_validator('Month')
    def validate_month(cls, v):
        valid_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        if v not in valid_months:
            raise ValueError(f'Month must be one of {valid_months}')
        return v
    
    @field_validator('VisitorType')
    def validate_visitor_type(cls, v):
        valid_types = ['Returning_Visitor', 'New_Visitor', 'Other']
        if v not in valid_types:
            raise ValueError(f'VisitorType must be one of {valid_types}')
        return v

class PredictionResponse(BaseModel):
    """
    Prediction response model
    """
    prediction: str
    probability: float
    timestamp: str

class ErrorResponse(BaseModel):
    """
    Error response model
    """
    error: str
    detail: Optional[str] = None
