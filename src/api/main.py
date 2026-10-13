from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    PredictRequest,
    PredictResponse,
    ExperimentRequest,
    ExperimentResponse,
    HealthResponse,
    ModelType
)
from src.api.model_loader import ModelManager

# Global model manager
model_manager: ModelManager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_manager
    print("[Startup] Initializing ModelManager and loading models...")
    model_manager = ModelManager()
    yield
    print("[Shutdown] Cleaning up...")

app = FastAPI(
    title="Customer Support Ticket Classification API",
    description="Production ML API for categorizing customer support tickets into billing, account, cancellation, product_inquiry, and refund.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for standard web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Info"])
def root():
    return {
        "service": "Customer Support Ticket Classifier",
        "docs_url": "/docs",
        "health_check": "/health",
        "best_model": "logistic_regression"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    if not model_manager or not model_manager.models:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models are not initialized"
        )
    return HealthResponse(
        status="healthy",
        available_models=model_manager.available_models,
        best_model="logistic_regression"
    )

@app.post("/predict", response_model=PredictResponse, tags=["Production"])
def predict(request: PredictRequest):
    """
    Production Route: Uses the best-performing model (Logistic Regression, 98.5% Accuracy)
    to classify support ticket text.
    """
    clean_text = request.text.strip()
    if not clean_text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Ticket text cannot be empty or just whitespace."
        )

    try:
        category, confidence = model_manager.predict(clean_text, model_name="logistic_regression")
        return PredictResponse(category=category, confidence=confidence)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )

@app.post("/experiment", response_model=ExperimentResponse, tags=["Experiments"])
def experiment(request: ExperimentRequest):
    """
    Advanced Route: Allows the caller to select which trained model to evaluate:
    - logistic_regression
    - decision_tree
    - random_forest
    - xgboost
    """
    clean_text = request.text.strip()
    if not clean_text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Ticket text cannot be empty or just whitespace."
        )

    model_key = request.model.value
    if model_key not in model_manager.available_models:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Requested model '{model_key}' is not available. Loaded: {model_manager.available_models}"
        )

    try:
        category, confidence = model_manager.predict(clean_text, model_name=model_key)
        return ExperimentResponse(
            category=category,
            confidence=confidence,
            model_used=model_key
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )
