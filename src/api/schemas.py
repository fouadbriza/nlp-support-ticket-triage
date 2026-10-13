from enum import Enum
from pydantic import BaseModel, Field

class ModelType(str, Enum):
    LOGISTIC_REGRESSION = "logistic_regression"
    DECISION_TREE = "decision_tree"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"

class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="Customer ticket text to classify",
        json_schema_extra={"example": "Why is there an unexpected extra fee on my monthly bill?"}
    )

class PredictResponse(BaseModel):
    category: str = Field(..., description="Predicted support category")
    confidence: float = Field(..., description="Prediction confidence score between 0.0 and 1.0")

class ExperimentRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="Customer ticket text to classify",
        json_schema_extra={"example": "Why is there an unexpected extra fee on my monthly bill?"}
    )
    model: ModelType = Field(default=ModelType.LOGISTIC_REGRESSION, description="Model to use for inference")

class ExperimentResponse(BaseModel):
    category: str = Field(..., description="Predicted support category")
    confidence: float = Field(..., description="Prediction confidence score between 0.0 and 1.0")
    model_used: str = Field(..., description="Model identifier used for inference")

class HealthResponse(BaseModel):
    status: str
    available_models: list[str]
    best_model: str
