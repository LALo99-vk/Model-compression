"""
Model Router - Handles model selection and configuration
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import json
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    with open("models/selected_model_config.json", "w", encoding="utf-8") as f:
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
    Uses exact file from comparison report to ensure size matches UI
    """
    # First, try to get exact path from comparison report
    comparison_report_path = "results/compression_comparison_report.json"
    if os.path.exists(comparison_report_path):
        try:
            with open(comparison_report_path, "r") as f:
                report = json.load(f)
            # Check if original model path is stored
            original_path = report.get("original", {}).get("model_path")
            if original_path and os.path.exists(original_path):
                _, ext = os.path.splitext(original_path)
                filename = f"original_model{ext}"
                file_size_mb = os.path.getsize(original_path) / (1024 * 1024)
                logger.info(f"✅ Serving original model from report: {filename} ({file_size_mb:.4f} MB)")
                return FileResponse(
                    path=original_path,
                    media_type="application/octet-stream",
                    filename=filename
                )
        except Exception as e:
            logger.warning(f"Could not read comparison report: {e}")
    
    # Fallback: Check for model files in priority order
    possible_files = [
        ("models/original_model.pt", "application/octet-stream", "original_model.pt"),
        ("models/original_model.pkl", "application/octet-stream", "original_model.pkl"),
        ("models/original_model.h5", "application/octet-stream", "original_model.h5"),
    ]
    
    for file_path, media_type, filename in possible_files:
        if os.path.exists(file_path):
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            logger.info(f"✅ Serving {filename} ({file_size_mb:.4f} MB)")
            return FileResponse(
                path=file_path,
                media_type=media_type,
                filename=filename
            )
    
    raise HTTPException(
        status_code=404,
        detail="No trained model found. Please train a model first. Expected: original_model.pt or original_model.pkl"
    )


@router.get("/trained")
async def get_trained_models():
    """
    Get all trained models with their details
    Returns model type, dataset used, original and compressed info
    Reads from training history to show multiple sessions
    """
    result = {
        "sessions": [],
        "count": 0
    }
    
    history_path = "results/training_history.json"
    compression_path = "results/compression_comprehensive.json"
    compression_report_path = "results/compression_comparison_report.json"
    
    # Try to load from training history first
    if os.path.exists(history_path):
        try:
            with open(history_path, "r") as f:
                history = json.load(f)
            
            # Load compression info for the latest session
            compressed_info = None
            if os.path.exists(compression_path):
                try:
                    with open(compression_path, "r") as f:
                        comp_data = json.load(f)
                        if comp_data.get("best_model"):
                            best = comp_data["best_model"]
                            compressed_info = {
                                "accuracy": (best.get("accuracy", 0) * 100) if best.get("accuracy", 0) <= 1 else best.get("accuracy", 0),
                                "size_kb": best.get("size_kb", best.get("compressed_size_kb", 0)),
                                "parameters": best.get("parameters", best.get("compressed_parameters", 0)),
                                "compression_ratio": best.get("compression_ratio", 1),
                                "size_reduction": best.get("size_reduction_percent", 0),
                                "method": best.get("method", "Best"),
                                "path": best.get("path", "models/compressed_model.pt")
                            }
                except Exception as e:
                    logger.warning(f"Could not load compression info: {e}")
            
            # Fallback to comparison report
            if not compressed_info and os.path.exists(compression_report_path):
                try:
                    with open(compression_report_path, "r") as f:
                        report = json.load(f)
                        if report.get("compressed"):
                            comp = report["compressed"]
                            compressed_info = {
                                "accuracy": (comp.get("metrics", {}).get("accuracy", 0) * 100) if comp.get("metrics", {}).get("accuracy", 0) <= 1 else comp.get("metrics", {}).get("accuracy", 0),
                                "size_kb": comp.get("size_kb", comp.get("size_mb", 0) * 1024),
                                "parameters": comp.get("parameters", 0),
                                "compression_ratio": report.get("compression_ratio", 1),
                                "size_reduction": report.get("reduction_percent", 0),
                                "method": comp.get("method", "Compressed"),
                                "path": comp.get("model_path", "models/compressed_model.pt")
                            }
                except Exception as e:
                    logger.warning(f"Could not load comparison report: {e}")
            
            # Process each session from history
            for i, session in enumerate(history):
                # Convert stored format to API format
                original = session.get("original", {})
                accuracy = original.get("accuracy", 0)
                if accuracy <= 1:
                    accuracy = accuracy * 100
                
                processed_session = {
                    "id": session.get("id", f"session_{i}"),
                    "model_type": session.get("model_type", "unknown"),
                    "dataset_name": session.get("dataset_name", "Unknown"),
                    "dataset_path": session.get("dataset_path", ""),
                    "created_at": session.get("created_at"),
                    "training_time": session.get("training_time", 0),
                    "original": {
                        "accuracy": accuracy,
                        "size_kb": original.get("size_kb", original.get("size_mb", 0) * 1024),
                        "parameters": original.get("parameters", 0),
                        "path": original.get("path", "")
                    },
                    "compressed": None
                }
                
                # Only add compression info to the first (latest) session
                # since compression overwrites the previous compressed model
                if i == 0 and compressed_info:
                    # Calculate size reduction if not provided
                    if compressed_info.get("size_reduction", 0) == 0:
                        if processed_session["original"]["size_kb"] > 0 and compressed_info["size_kb"] > 0:
                            compressed_info["size_reduction"] = (
                                (processed_session["original"]["size_kb"] - compressed_info["size_kb"]) 
                                / processed_session["original"]["size_kb"] * 100
                            )
                    processed_session["compressed"] = compressed_info
                
                result["sessions"].append(processed_session)
            
            result["count"] = len(result["sessions"])
            return result
            
        except Exception as e:
            logger.error(f"Error loading training history: {e}")
    
    # Fallback to loading from current training logs (for backward compatibility)
    logs_path = "results/training_logs.json"
    config_path = "models/selected_model_config.json"
    
    if not os.path.exists(logs_path):
        return result
    
    try:
        with open(logs_path, "r") as f:
            training_logs = json.load(f)
        
        dataset_path = ""
        dataset_name = "Unknown"
        
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
                dataset_path = config.get("dataset_path", "")
                dataset_name = dataset_path.split("/")[-1] if dataset_path else "Unknown"
        
        original_size_kb = training_logs.get("model_size_kb", training_logs.get("model_size_mb", 0) * 1024)
        original_accuracy = training_logs.get("val_score", training_logs.get("train_score", 0))
        if training_logs.get("history") and len(training_logs["history"]) > 0:
            original_accuracy = training_logs["history"][-1].get("val_accuracy", original_accuracy)
        
        session = {
            "id": "current",
            "model_type": training_logs.get("model_type", "unknown"),
            "dataset_name": dataset_name,
            "dataset_path": dataset_path,
            "created_at": None,
            "training_time": training_logs.get("training_time", 0),
            "original": {
                "accuracy": original_accuracy * 100 if original_accuracy <= 1 else original_accuracy,
                "size_kb": original_size_kb,
                "parameters": training_logs.get("total_parameters", training_logs.get("num_parameters", 0)),
                "path": training_logs.get("model_path", "")
            },
            "compressed": None
        }
        
        result["sessions"].append(session)
        result["count"] = 1
        
    except Exception as e:
        logger.error(f"Error loading trained models: {e}")
    
    return result


@router.get("/download/compressed")
async def download_compressed_model():
    """
    Download the compressed model
    Uses exact file from comparison report to ensure size matches UI
    """
    # First, try to get exact path from comparison report
    comparison_report_path = "results/compression_comparison_report.json"
    if os.path.exists(comparison_report_path):
        try:
            with open(comparison_report_path, "r") as f:
                report = json.load(f)
            # Check if compressed model path is stored
            compressed_path = report.get("compressed", {}).get("model_path")
            if compressed_path and os.path.exists(compressed_path):
                _, ext = os.path.splitext(compressed_path)
                filename = f"compressed_model{ext}"
                file_size_mb = os.path.getsize(compressed_path) / (1024 * 1024)
                logger.info(f"✅ Serving compressed model from report: {filename} ({file_size_mb:.4f} MB)")
                return FileResponse(
                    path=compressed_path,
                    media_type="application/octet-stream",
                    filename=filename
                )
        except Exception as e:
            logger.warning(f"Could not read comparison report: {e}")
    
    # Fallback: Check for compressed model files
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
            logger.info(f"✅ Serving {filename} ({file_size_mb:.4f} MB)")
            return FileResponse(
                path=file_path,
                media_type=media_type,
                filename=filename
            )
    
    raise HTTPException(
        status_code=404,
        detail="No compressed model found. Please compress a model first."
    )