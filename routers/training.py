"""
Training Router - Handles model training
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
import json
import os
from services.training_service import TrainingService

router = APIRouter()
training_service = TrainingService()


class TrainingRequest(BaseModel):
    dataset_path: str
    epochs: Optional[int] = 10
    batch_size: Optional[int] = 32
    validation_split: Optional[float] = 0.2


@router.post("/start")
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """
    Start training the selected model
    """
    # Check if model is selected
    config_path = "models/selected_model_config.json"
    if not os.path.exists(config_path):
        raise HTTPException(
            status_code=400,
            detail="No model selected. Please select a model first."
        )

    # Check if dataset exists
    if not os.path.exists(request.dataset_path):
        raise HTTPException(
            status_code=404,
            detail=f"Dataset not found: {request.dataset_path}"
        )

    # Load model config and save dataset path
    with open(config_path, "r") as f:
        model_config = json.load(f)
    
    # Save dataset path for compression use
    model_config["dataset_path"] = request.dataset_path
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=2)

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
        "status": "training"
    }


@router.get("/status")
async def get_training_status():
    """Get current training status"""
    status_path = "results/training_status.json"

    if not os.path.exists(status_path):
        return {
            "status": "not_started",
            "message": "No training in progress"
        }

    with open(status_path, "r") as f:
        status = json.load(f)

    return status


@router.get("/logs")
async def get_training_logs():
    """Get training logs"""
    logs_path = "results/training_logs.json"

    if not os.path.exists(logs_path):
        raise HTTPException(
            status_code=404,
            detail="No training logs found"
        )

    with open(logs_path, "r") as f:
        logs = json.load(f)

    return logs


@router.delete("/stop")
async def stop_training():
    """Stop current training"""
    training_service.stop_training()
    return {"message": "Training stopped"}