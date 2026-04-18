"""Pydantic request/response schemas for the churn prediction API."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class ContractType(str, Enum):
    month_to_month = "Month-to-month"
    one_year = "One year"
    two_year = "Two year"


class InternetService(str, Enum):
    dsl = "DSL"
    fiber_optic = "Fiber optic"
    no = "No"


class PaymentMethod(str, Enum):
    electronic_check = "Electronic check"
    mailed_check = "Mailed check"
    bank_transfer = "Bank transfer (automatic)"
    credit_card = "Credit card (automatic)"


class CustomerFeatures(BaseModel):
    """Input schema for a single customer churn prediction request."""

    gender: str = Field(..., pattern="^(Male|Female)$", description="Customer gender")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="Is the customer a senior citizen (0/1)")
    Partner: str = Field(..., pattern="^(Yes|No)$")
    Dependents: str = Field(..., pattern="^(Yes|No)$")
    tenure: int = Field(..., ge=0, le=120, description="Months as customer (0–120)")
    PhoneService: str = Field(..., pattern="^(Yes|No)$")
    MultipleLines: str = Field(..., pattern="^(Yes|No|No phone service)$")
    InternetService: InternetService
    OnlineSecurity: str = Field(..., pattern="^(Yes|No|No internet service)$")
    OnlineBackup: str = Field(..., pattern="^(Yes|No|No internet service)$")
    DeviceProtection: str = Field(..., pattern="^(Yes|No|No internet service)$")
    TechSupport: str = Field(..., pattern="^(Yes|No|No internet service)$")
    StreamingTV: str = Field(..., pattern="^(Yes|No|No internet service)$")
    StreamingMovies: str = Field(..., pattern="^(Yes|No|No internet service)$")
    Contract: ContractType
    PaperlessBilling: str = Field(..., pattern="^(Yes|No)$")
    PaymentMethod: PaymentMethod
    MonthlyCharges: float = Field(..., ge=0.0, le=500.0, description="Monthly bill in USD")
    TotalCharges: float = Field(..., ge=0.0, description="Total billed to date in USD")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 12,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 70.35,
            "TotalCharges": 843.40,
        }
    })


class PredictionResponse(BaseModel):
    """Churn prediction result for a single customer."""

    churn_probability: float = Field(..., description="Probability of churn in [0, 1]")
    churn_label: bool = Field(..., description="True if predicted to churn")
    confidence: str = Field(..., description="HIGH / MEDIUM / LOW based on distance from 0.5")
    model_name: str
    model_version: str
    prediction_id: str


class HealthResponse(BaseModel):
    """API health check response."""

    status: str
    model_name: str
    model_version: str
    mlflow_tracking_uri: str


class ModelInfoResponse(BaseModel):
    """Metadata and metrics of the currently loaded Production model."""

    model_name: str
    model_version: str
    run_id: str
    metrics: dict[str, float]
    parameters: dict[str, str]
    registered_at: str | None
