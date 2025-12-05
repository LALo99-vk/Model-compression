"""
Model Router - Handles model selection and configuration
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
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


@router.get("/download/original")
async def download_original_model():
    """
    Download the original trained model
    Auto-detects file type (.pt for PyTorch, .pkl for scikit-learn)
    """
    # Check for model files in priority order
    possible_files = [
        ("models/original_model.pt", "application/octet-stream", "original_model.pt"),
        ("models/original_model.pkl", "application/octet-stream", "original_model.pkl"),
        ("models/original_model.h5", "application/octet-stream", "original_model.h5"),
    ]
    
    for file_path, media_type, filename in possible_files:
        if os.path.exists(file_path):
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"✅ Serving {filename} ({file_size_mb:.4f} MB)")
            return FileResponse(
                path=file_path,
                media_type=media_type,
                filename=filename
            )
    
    raise HTTPException(
        status_code=404,
        detail="No trained model found. Please train a model first. Expected: original_model.pt or original_model.pkl"
    )


@router.get("/download/compressed")
async def download_compressed_model():
    """
    Download the compressed model
    Auto-detects file type (.pt for PyTorch, .pkl for scikit-learn)
    """
    # Check for compressed model files
    possible_files = [
        ("models/compressed_model.pt", "application/octet-stream", "compressed_model.pt"),
        ("models/compressed_model.pkl", "application/octet-stream", "compressed_model.pkl"),
        ("models/quantized_model.pt", "application/octet-stream", "quantized_model.pt"),
        ("models/pruned_model.pt", "application/octet-stream", "pruned_model.pt"),
        ("models/distilled_model.pt", "application/octet-stream", "distilled_model.pt"),
    ]
    
    for file_path, media_type, filename in possible_files:
        if os.path.exists(file_path):
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"✅ Serving {filename} ({file_size_mb:.4f} MB)")
            return FileResponse(
                path=file_path,
                media_type=media_type,
                filename=filename
            )
    
    raise HTTPException(
        status_code=404,
        detail="No compressed model found. Please compress a model first."
    )