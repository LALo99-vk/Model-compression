"""
Compression Router - Handles model compression
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import json
import os
from services.compression_service import CompressionService

router = APIRouter()
compression_service = CompressionService()


class CompressionRequest(BaseModel):
    method: str  # 'pruning', 'quantization', 'distillation'
    pruning_amount: Optional[float] = 0.3
    quantization_bits: Optional[int] = 8
    distillation_temperature: Optional[float] = 3.0
    distillation_alpha: Optional[float] = 0.5


@router.post("/compress")
async def compress_model(request: CompressionRequest):
    """
    Compress the trained model using specified method
    """
    method = request.method.lower()

    if method not in ["pruning", "quantization", "distillation"]:
        raise HTTPException(
            status_code=400,
            detail="Method must be 'pruning', 'quantization', or 'distillation'"
        )

    # Check if original model exists
    possible_paths = [
        "models/original_model.pt",
        "models/original_model.h5", 
        "models/original_model.pkl"
    ]
    
    original_model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            original_model_path = path
            break
    
    if original_model_path is None:
        raise HTTPException(
            status_code=404,
            detail="Original model not found. Please train a model first."
        )

    # Perform compression
    result = compression_service.compress(
        original_model_path,
        method=method,
        pruning_amount=request.pruning_amount,
        quantization_bits=request.quantization_bits,
        distillation_temperature=request.distillation_temperature,
        distillation_alpha=request.distillation_alpha
    )

    # Save compression info
    compression_info = {
        "method": method,
        "parameters": {
            "pruning_amount": request.pruning_amount,
            "quantization_bits": request.quantization_bits,
            "distillation_temperature": request.distillation_temperature,
            "distillation_alpha": request.distillation_alpha
        },
        "result": result
    }

    with open("results/compression_info.json", "w") as f:
        json.dump(compression_info, f, indent=2)

    return {
        "message": "Model compressed successfully",
        "method": method,
        "compressed_model_path": f"models/compressed_model{os.path.splitext(original_model_path)[1]}",
        "compression_info": compression_info
    }


@router.get("/methods")
async def get_compression_methods():
    """Get available compression methods and their descriptions"""
    methods = {
        "pruning": {
            "name": "Weight Pruning",
            "description": "Remove weights with small magnitudes",
            "parameters": {
                "pruning_amount": "Percentage of weights to prune (0.0-1.0)"
            }
        },
        "quantization": {
            "name": "Quantization",
            "description": "Reduce precision of weights",
            "parameters": {
                "quantization_bits": "Number of bits for quantization (4, 8, 16)"
            }
        },
        "distillation": {
            "name": "Knowledge Distillation",
            "description": "Train smaller model to mimic larger model",
            "parameters": {
                "distillation_temperature": "Temperature for softmax (1.0-10.0)",
                "distillation_alpha": "Weight for distillation loss (0.0-1.0)"
            }
        }
    }

    return {"methods": methods}


@router.get("/info")
async def get_compression_info():
    """Get information about the last compression"""
    info_path = "results/compression_info.json"

    if not os.path.exists(info_path):
        raise HTTPException(
            status_code=404,
            detail="No compression information found"
        )

    with open(info_path, "r") as f:
        info = json.load(f)

    return info