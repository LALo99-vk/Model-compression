"""
Evaluation Router - Handles model evaluation
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import json
import os
from services.evaluation_service import EvaluationService

router = APIRouter(tags=["Evaluation"])

evaluation_service = EvaluationService()


class EvaluationRequest(BaseModel):
    model_type: str  # 'original' or 'compressed'
    dataset_path: str


@router.post("/evaluate")
async def evaluate_model(request: EvaluationRequest):
    """
    Evaluate original or compressed model
    """
    # Determine model path based on model_type
    if request.model_type == "original":
        # Check for original model with different extensions
        possible_paths = [
            f"models/original_model.pkl",
            f"models/original_model.pt", 
            f"models/original_model.h5"
        ]
    else:
        # Check for compressed models or other model types
        possible_paths = [
            f"models/{request.model_type}_model.pkl",
            f"models/{request.model_type}_model.pt",
            f"models/{request.model_type}_model.h5"
        ]
    
    # Find the first existing model file
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        raise HTTPException(
            status_code=404,
            detail=f"Model not found. Checked paths: {possible_paths}"
        )

    # Check if dataset exists
    if not os.path.exists(request.dataset_path):
        raise HTTPException(
            status_code=404,
            detail=f"Dataset not found: {request.dataset_path}"
        )

    # Perform evaluation
    metrics = evaluation_service.evaluate(
        model_path,
        request.dataset_path,
        request.model_type
    )

    # Save metrics
    metrics_path = f"results/{request.model_type}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return {
        "message": "Evaluation completed",
        "model_type": request.model_type,
        "metrics": metrics,
        "saved_to": metrics_path
    }


@router.get("/metrics/{model_type}")
async def get_metrics(model_type: str):
    """
    Get saved metrics for original or compressed model
    """
    if model_type not in ["original", "compressed"]:
        raise HTTPException(
            status_code=400,
            detail="model_type must be 'original' or 'compressed'"
        )

    metrics_path = f"results/{model_type}_metrics.json"

    if not os.path.exists(metrics_path):
        raise HTTPException(
            status_code=404,
            detail=f"Metrics not found for {model_type} model"
        )

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    return metrics


@router.get("/all-metrics")
async def get_all_metrics():
    """Get metrics for both original and compressed models"""
    original_path = "results/original_metrics.json"
    compressed_path = "results/compressed_metrics.json"

    result = {}

    if os.path.exists(original_path):
        with open(original_path, "r") as f:
            result["original"] = json.load(f)

    if os.path.exists(compressed_path):
        with open(compressed_path, "r") as f:
            result["compressed"] = json.load(f)

    if not result:
        raise HTTPException(
            status_code=404,
            detail="No evaluation metrics found"
        )

    return result