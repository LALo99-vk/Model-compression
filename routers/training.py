"""
Training Router - Handles model training with comprehensive validation
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional
import json
import os
import logging
from services.training_service import TrainingService
from utils.validation import DataValidator
from services.dataset_validation_service import DatasetValidationService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()
training_service = TrainingService()
validator = DataValidator()
dataset_validator = DatasetValidationService()


class TrainingRequest(BaseModel):
    dataset_path: str = Field(..., description="Path to the dataset file")
    epochs: Optional[int] = Field(10, ge=1, le=1000, description="Number of training epochs")
    batch_size: Optional[int] = Field(32, ge=1, le=1024, description="Batch size for training")
    validation_split: Optional[float] = Field(0.2, ge=0.1, le=0.5, description="Validation split ratio")


@router.post("/start")
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """
    Start training the selected model with comprehensive validation
    """
    try:
        # Validate parameters
        if request.validation_split <= 0 or request.validation_split >= 1:
            raise HTTPException(
                status_code=400,
                detail=f"validation_split must be between 0 and 1, got {request.validation_split}"
            )
        
        if request.epochs <= 0:
            raise HTTPException(
                status_code=400,
                detail=f"epochs must be positive, got {request.epochs}"
            )
        
        # Check if model is selected
        config_path = "models/selected_model_config.json"
        if not os.path.exists(config_path):
            raise HTTPException(
                status_code=400,
                detail="No model selected. Please select a model first."
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

        # Load model config first to get model_type
        try:
            with open(config_path, "r") as f:
                model_config = json.load(f)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=500,
                detail="Model configuration file is corrupted. Please select a model again."
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error reading model configuration: {str(e)}"
            )
        
        # Validate model config
        if 'model_type' not in model_config:
            raise HTTPException(
                status_code=400,
                detail="Invalid model configuration: model_type missing"
            )
        
        # 🚦 PHASE 1: Check dataset validation BEFORE starting training
        validation_result = dataset_validator.validate_dataset(
            request.dataset_path,
            model_config['model_type']
        )
        
        if not validation_result['is_valid']:
            # Return JSON error - NEVER start training if invalid
            raise HTTPException(
                status_code=400,
                detail=json.dumps({
                    "status": "invalid",
                    "message": "Dataset validation failed. Please validate and fix the dataset first.",
                    "issues": validation_result.get('issues', []),
                    "can_autofix": validation_result.get('can_auto_fix', False)
                })
            )
    
        # Save dataset path for compression use
        model_config["dataset_path"] = request.dataset_path
        os.makedirs("models", exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(model_config, f, indent=2)

        logger.info(f"✅ Dataset validated. Starting training: model={model_config.get('model_type')}, dataset={request.dataset_path}, epochs={request.epochs}")

        # Start training in background
        background_tasks.add_task(
            training_service.train_model,
            model_config,
            request.dataset_path,
            request.epochs,
            request.batch_size,
            request.validation_split
        )

        return {
            "message": "Training started",
            "model_type": model_config["model_type"],
            "dataset": request.dataset_path,
            "epochs": request.epochs,
            "batch_size": request.batch_size,
            "validation_split": request.validation_split,
            "status": "training"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error starting training: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start training: {str(e)}"
        )


@router.get("/status")
async def get_training_status():
    """Get current training status"""
    status_path = "results/training_status.json"

    if not os.path.exists(status_path):
        return {
            "status": "not_started",
            "message": "No training in progress"
        }

    try:
        with open(status_path, "r") as f:
            status = json.load(f)
        return status
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="Training status file is corrupted"
        )
    except Exception as e:
        logger.error(f"Error reading training status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error reading training status: {str(e)}"
        )


@router.get("/logs")
async def get_training_logs():
    """Get training logs"""
    logs_path = "results/training_logs.json"

    if not os.path.exists(logs_path):
        raise HTTPException(
            status_code=404,
            detail="No training logs found. Training may not have started yet."
        )

    try:
        with open(logs_path, "r") as f:
            logs = json.load(f)
        return logs
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="Training logs file is corrupted"
        )
    except Exception as e:
        logger.error(f"Error reading training logs: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error reading training logs: {str(e)}"
        )


@router.delete("/stop")
async def stop_training():
    """Stop current training"""
    try:
        training_service.stop_training()
        return {"message": "Training stop signal sent"}
    except Exception as e:
        logger.error(f"Error stopping training: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error stopping training: {str(e)}"
        )
