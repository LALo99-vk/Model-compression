"""
Evaluation Router - Handles model evaluation with comprehensive validation
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import json
import os
import logging
from services.evaluation_service import EvaluationService
from utils.validation import DataValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(tags=["Evaluation"])

evaluation_service = EvaluationService()
validator = DataValidator()


class EvaluationRequest(BaseModel):
    model_type: str = Field(..., description="Model type: 'original' or 'compressed'")
    dataset_path: str = Field(..., description="Path to the evaluation dataset")


@router.post("/evaluate")
async def evaluate_model(request: EvaluationRequest):
    """
    Evaluate original or compressed model with comprehensive validation
    """
    try:
        # Validate model_type
        if request.model_type not in ["original", "compressed"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model_type: {request.model_type}. Must be 'original' or 'compressed'"
            )
        
        # Validate dataset path
        try:
            validator.validate_dataset_path(request.dataset_path)
        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail=f"Dataset not found: {request.dataset_path}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid dataset path: {str(e)}"
            )
        
        # Determine model path based on model_type
        if request.model_type == "original":
            # Check for original model with different extensions
            possible_paths = [
                "models/original_model.pkl",
                "models/original_model.pt", 
                "models/original_model.h5"
            ]
        else:
            # Check for compressed models
            possible_paths = [
                "models/compressed_model.pkl",
                "models/compressed_model.pt",
                "models/compressed_model.h5"
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
                detail=f"{request.model_type.capitalize()} model not found. Please train/compress the model first. Checked paths: {possible_paths}"
            )

        logger.info(f"Evaluating {request.model_type} model on dataset: {request.dataset_path}")

        # Perform evaluation with comprehensive error handling
        try:
            metrics = evaluation_service.evaluate(
                model_path,
                request.dataset_path,
                request.model_type
            )
        except ValueError as e:
            error_msg = str(e)
            # Provide helpful error messages for common issues
            if "Feature count mismatch" in error_msg:
                raise HTTPException(
                    status_code=400,
                    detail=f"{error_msg}. Please use the same dataset that was used for training, or re-train the model."
                )
            elif "task type" in error_msg.lower() or "classification" in error_msg.lower() or "regression" in error_msg.lower():
                raise HTTPException(
                    status_code=400,
                    detail=f"Task type mismatch: {error_msg}. The model may have been trained with a different task type. Please re-train the model."
                )
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Evaluation error: {error_msg}"
                )

        # Save metrics
        os.makedirs("results", exist_ok=True)
        metrics_path = f"results/{request.model_type}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        logger.info(f"Evaluation completed successfully for {request.model_type} model. Task type: {metrics.get('task_type', 'unknown')}")

        return {
            "message": "Evaluation completed",
            "model_type": request.model_type,
            "task_type": metrics.get('task_type', 'unknown'),
            "metrics": metrics,
            "saved_to": metrics_path
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {str(e)}. Please check that the model was trained correctly and the dataset is compatible."
        )


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
