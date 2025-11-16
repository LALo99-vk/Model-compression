"""
Model Router - Handles model selection and configuration
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import json
import os

router = APIRouter()


class ModelSelection(BaseModel):
    model_type: str  # 'cnn', 'rnn', 'decision_tree'
    task_type: str  # 'classification', 'regression'
    input_shape: Optional[tuple] = None
    num_classes: Optional[int] = None
    config: Optional[Dict[str, Any]] = None


AVAILABLE_MODELS = {
    "cnn": {
        "name": "Convolutional Neural Network",
        "description": "CNN for image classification",
        "supported_tasks": ["classification"],
        "default_config": {
            "conv_layers": 3,
            "filters": [32, 64, 128],
            "kernel_size": 3,
            "pool_size": 2,
            "dense_units": 128,
            "dropout": 0.5,
            "learning_rate": 0.001
        }
    },
    "rnn": {
        "name": "Recurrent Neural Network",
        "description": "RNN/LSTM for sequence data",
        "supported_tasks": ["classification", "regression"],
        "default_config": {
            "rnn_type": "LSTM",
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.3,
            "bidirectional": True,
            "learning_rate": 0.001
        }
    },
    "decision_tree": {
        "name": "Decision Tree",
        "description": "Tree-based model for tabular data",
        "supported_tasks": ["classification", "regression"],
        "default_config": {
            "max_depth": 10,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "criterion": "gini"
        }
    }
}


@router.get("/available")
async def get_available_models():
    """Get list of available models"""
    return {
        "models": AVAILABLE_MODELS,
        "count": len(AVAILABLE_MODELS)
    }


@router.post("/select")
async def select_model(selection: ModelSelection):
    """
    Select and configure a model
    """
    model_type = selection.model_type.lower()

    if model_type not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model type. Available: {list(AVAILABLE_MODELS.keys())}"
        )

    model_info = AVAILABLE_MODELS[model_type]

    if selection.task_type not in model_info["supported_tasks"]:
        raise HTTPException(
            status_code=400,
            detail=f"Task {selection.task_type} not supported for {model_type}"
        )

    # Merge default config with user config
    config = model_info["default_config"].copy()
    if selection.config:
        config.update(selection.config)

    # Save selection
    model_config = {
        "model_type": model_type,
        "task_type": selection.task_type,
        "input_shape": selection.input_shape,
        "num_classes": selection.num_classes,
        "config": config,
        "model_info": model_info
    }

    # Save to file
    with open("models/selected_model_config.json", "w") as f:
        json.dump(model_config, f, indent=2)

    return {
        "message": "Model selected successfully",
        "selection": model_config
    }


@router.get("/current")
async def get_current_selection():
    """Get currently selected model configuration"""
    config_path = "models/selected_model_config.json"

    if not os.path.exists(config_path):
        raise HTTPException(
            status_code=404,
            detail="No model selected. Please select a model first."
        )

    with open(config_path, "r") as f:
        config = json.load(f)

    return config