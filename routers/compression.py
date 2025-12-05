"""
Compression Router - Handles model compression with comprehensive validation
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import json
import os
import logging
from services.compression_service import CompressionService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()
compression_service = CompressionService()


class CompressionRequest(BaseModel):
    method: str = Field(..., description="Compression method: 'pruning', 'quantization', 'distillation', or 'comprehensive'")
    pruning_amount: Optional[float] = Field(0.3, ge=0.0, le=1.0, description="Pruning amount (0.0-1.0)")
    quantization_bits: Optional[int] = Field(8, ge=4, le=16, description="Quantization bits (4, 8, or 16)")
    distillation_temperature: Optional[float] = Field(3.0, ge=1.0, le=10.0, description="Distillation temperature")
    distillation_alpha: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Distillation alpha")


class ComprehensiveCompressionRequest(BaseModel):
    pruning_amount: Optional[float] = Field(0.35, ge=0.2, le=0.5, description="Pruning amount (0.2-0.5, 20-50%)")
    quantization_bits: Optional[int] = Field(8, ge=4, le=16, description="Quantization bits (4, 8, or 16)")
    distillation_temperature: Optional[float] = Field(3.0, ge=1.0, le=10.0, description="Distillation temperature")
    distillation_alpha: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Distillation alpha")


@router.post("/compress")
async def compress_model(request: CompressionRequest):
    """
    Compress the trained model using specified method with comprehensive validation
    Supports single method or 'comprehensive' to apply all three techniques
    """
    try:
        method = request.method.lower()

        # Check for comprehensive compression
        if method == "comprehensive":
            return await compress_comprehensive(
                ComprehensiveCompressionRequest(
                    pruning_amount=request.pruning_amount,
                    quantization_bits=request.quantization_bits,
                    distillation_temperature=request.distillation_temperature,
                    distillation_alpha=request.distillation_alpha
                )
            )

        if method not in ["pruning", "quantization", "distillation"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid compression method: {method}. Must be 'pruning', 'quantization', 'distillation', or 'comprehensive'"
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
                detail="Original model not found. Please train a model first. Expected paths: " + ", ".join(possible_paths)
            )

        logger.info(f"Starting compression with method: {method}")

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
        os.makedirs("results", exist_ok=True)
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
            json.dump(compression_info, f, indent=2, default=str)

        logger.info(f"Compression completed successfully: {method}")

        return {
            "message": "Model compressed successfully",
            "method": method,
            "compressed_model_path": f"models/compressed_model{os.path.splitext(original_model_path)[1]}",
            "compression_info": compression_info
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during compression: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Compression failed: {str(e)}"
        )


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
    """Get information about the last compression and original model info from training"""
    info_path = "results/compression_info.json"
    
    # Try to load compression info if available
    compression_info = None
    if os.path.exists(info_path):
        with open(info_path, "r") as f:
            compression_info = json.load(f)
    
    # Always try to get original model info from training logs
    original_model_info = None
    training_logs_path = "results/training_logs.json"
    training_status_path = "results/training_status.json"
    
    if os.path.exists(training_logs_path):
        try:
            with open(training_logs_path, "r") as f:
                logs = json.load(f)
                if logs.get("model_size_mb") or logs.get("model_size_bytes"):
                    original_model_info = {
                        "model_size_bytes": logs.get("model_size_bytes"),
                        "model_size_mb": logs.get("model_size_mb"),
                        "model_size_kb": logs.get("model_size_kb"),
                        "total_parameters": logs.get("total_parameters"),
                        "trainable_parameters": logs.get("trainable_parameters"),
                        "num_parameters": logs.get("num_parameters"),
                        "model_path": logs.get("model_path"),
                        "model_type": logs.get("model_type")
                    }
        except Exception as e:
            logger.warning(f"Could not read training logs: {str(e)}")
    
    # Also check training status for model size
    if not original_model_info and os.path.exists(training_status_path):
        try:
            with open(training_status_path, "r") as f:
                status = json.load(f)
                if status.get("model_size_mb") or status.get("model_size_bytes"):
                    original_model_info = {
                        "model_size_bytes": status.get("model_size_bytes"),
                        "model_size_mb": status.get("model_size_mb"),
                        "model_size_kb": status.get("model_size_kb"),
                        "total_parameters": status.get("total_parameters"),
                        "num_parameters": status.get("num_parameters"),
                        "model_path": status.get("model_path")
                    }
        except Exception as e:
            logger.warning(f"Could not read training status: {str(e)}")
    
    # If still no info, try to get from model file directly
    if not original_model_info:
        possible_paths = [
            "models/original_model.pt",
            "models/original_model.pkl",
            "models/original_model.h5"
        ]
        
        for model_path in possible_paths:
            if os.path.exists(model_path):
                model_size_bytes = os.path.getsize(model_path)
                original_model_info = {
                    "model_size_bytes": model_size_bytes,
                    "model_size_mb": round(model_size_bytes / (1024 * 1024), 4),
                    "model_size_kb": round(model_size_bytes / 1024, 2),
                    "model_path": model_path
                }
                break
    
    # If we have compression info, merge it with original model info
    if compression_info:
        result = compression_info.copy()
        if original_model_info and compression_info.get("result"):
            # Ensure original model size in compression result matches training logs
            if original_model_info.get("model_size_mb"):
                compression_info["result"]["original_size_mb"] = original_model_info["model_size_mb"]
            if original_model_info.get("model_size_bytes"):
                compression_info["result"]["original_size"] = original_model_info["model_size_bytes"]
            if original_model_info.get("total_parameters") or original_model_info.get("num_parameters"):
                compression_info["result"]["original_parameters"] = original_model_info.get("total_parameters") or original_model_info.get("num_parameters")
        return result
    elif original_model_info:
        # Return original model info even if compression hasn't been done
        return {
            "message": "Original model info from training",
            "original_model": original_model_info
        }
    else:
        raise HTTPException(
            status_code=404,
            detail="No compression information or trained model found"
        )


@router.get("/comparison")
async def get_compression_comparison():
    """
    Get comparison report between original and compressed models
    Returns data in format expected by frontend
    """
    try:
        # Priority 1: Load from compression comparison report
        comparison_report_path = "results/compression_comparison_report.json"
        if os.path.exists(comparison_report_path):
            with open(comparison_report_path, "r") as f:
                report = json.load(f)
            
            # Convert to frontend format with REAL DATA from compression report
            original_metrics = report.get("original", {}).get("metrics", {})
            compressed_metrics = report.get("compressed", {}).get("metrics", {})
            
            # Extract real accuracy values
            original_acc = original_metrics.get("accuracy", 0)
            compressed_acc = compressed_metrics.get("accuracy", 0)
            acc_diff = ((compressed_acc - original_acc) / original_acc * 100) if original_acc > 0 else 0
            
            # Extract real inference times
            original_inference = original_metrics.get("inference_time_ms", 0)
            compressed_inference = compressed_metrics.get("inference_time_ms", 0)
            speedup = (original_inference / compressed_inference) if compressed_inference > 0 else 1.0
            
            return {
                "file_size": {
                    "original_mb": report.get("original", {}).get("size_mb", 0),
                    "compressed_mb": report.get("compressed", {}).get("size_mb", 0),
                    "reduction_percent": report.get("reduction_percent", 0),
                    "compression_ratio": 1.0 / (1 - report.get("reduction_percent", 0) / 100) if report.get("reduction_percent", 0) > 0 else 1.0
                },
                "detailed_metrics": {
                    "original": {
                        "parameters": report.get("original", {}).get("parameters", 0),
                        "accuracy": original_acc,
                        "precision": original_metrics.get("precision", 0),
                        "recall": original_metrics.get("recall", 0),
                        "f1_score": original_metrics.get("f1_score", 0)
                    },
                    "compressed": {
                        "parameters": report.get("compressed", {}).get("parameters", 0),
                        "accuracy": compressed_acc,
                        "precision": compressed_metrics.get("precision", 0),
                        "recall": compressed_metrics.get("recall", 0),
                        "f1_score": compressed_metrics.get("f1_score", 0)
                    }
                },
                "accuracy": {
                    "original": original_acc,
                    "compressed": compressed_acc,
                    "difference_percent": acc_diff
                },
                "inference_time": {
                    "original_ms": original_inference,
                    "compressed_ms": compressed_inference,
                    "speedup": speedup
                }
            }
        
        # Fallback: Try old comprehensive results file
        comprehensive_path = "results/compression_comprehensive.json"
        if os.path.exists(comprehensive_path):
            with open(comprehensive_path, "r") as f:
                comparison_report = json.load(f)
                return comparison_report
        
        # Priority 2: Try to load from comprehensive compression results
        comprehensive_path = "results/compression_comprehensive.json"
        if os.path.exists(comprehensive_path):
            with open(comprehensive_path, "r") as f:
                comprehensive_data = json.load(f)
            
            comparison_report = comprehensive_data.get("comparison_report")
            if comparison_report:
                return comparison_report
        
        # Fallback: Try to generate comparison from compression info
        info_path = "results/compression_info.json"
        if os.path.exists(info_path):
            with open(info_path, "r") as f:
                compression_info = json.load(f)
            
            # Try to reconstruct comparison from compression info
            result = compression_info.get("result", {})
            if result:
                original_size_mb = result.get("original_size_mb", 0)
                compressed_size_mb = result.get("compressed_size_mb", 0)
                original_params = result.get("original_parameters", 0)
                compressed_params = result.get("compressed_parameters", 0)
                reduction_percent = result.get("size_reduction_percent", 0)
                
                # Load model config for architecture
                model_config_path = "models/original_model_arch.json"
                model_type = "Unknown"
                if os.path.exists(model_config_path):
                    with open(model_config_path, "r") as f:
                        arch_data = json.load(f)
                        config = arch_data.get("config", {})
                        model_type = config.get("model_type", "Unknown").upper()
                
                # Validate compression actually happened
                if reduction_percent < 0.1 or abs(original_size_mb - compressed_size_mb) < 0.001:
                    return {
                        "original": {
                            "size_mb": round(original_size_mb, 2),
                            "params": original_params,
                            "architecture": model_type
                        },
                        "compressed": {
                            "size_mb": round(compressed_size_mb, 2),
                            "params": compressed_params,
                            "architecture": f"{model_type} (Compressed)"
                        },
                        "reduction_percent": round(reduction_percent, 2),
                        "accuracy_drop": "N/A",
                        "success": False,
                        "failure_reason": "No real compression detected",
                        "guidance": "Compression produced negligible or no size reduction. Please retry compression."
                    }
                
                return {
                    "original": {
                        "size_mb": round(original_size_mb, 2),
                        "params": original_params,
                        "architecture": model_type
                    },
                    "compressed": {
                        "size_mb": round(compressed_size_mb, 2),
                        "params": compressed_params,
                        "architecture": f"{model_type} (Compressed)"
                    },
                    "reduction_percent": round(reduction_percent, 2),
                    "accuracy_drop": "N/A",
                    "success": True
                }
        
        raise HTTPException(
            status_code=404,
            detail="No compression comparison available. Please run compression first."
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating comparison report: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate comparison report: {str(e)}"
        )


@router.post("/compress/comprehensive")
async def compress_comprehensive(request: ComprehensiveCompressionRequest):
    """
    Apply all three compression techniques sequentially:
    1. Weight Pruning (20-50%)
    2. Quantization (FP32 → INT8)
    3. Knowledge Distillation (smaller student model)
    
    Returns all three compressed models + best performing one
    """
    try:
        logger.info("Starting comprehensive compression (all three methods)...")

        # Perform comprehensive compression
        result = compression_service.compress_comprehensive(
            pruning_amount=request.pruning_amount,
            quantization_bits=request.quantization_bits,
            distillation_temperature=request.distillation_temperature,
            distillation_alpha=request.distillation_alpha
        )

        # Check if compression failed
        if result.get("status") == "compression_failed":
            raise HTTPException(
                status_code=400,
                detail=f"Compression failed: {result.get('reason', 'Unknown error')}. Details: {result.get('details', '')}"
            )

        # Save comprehensive compression info
        os.makedirs("results", exist_ok=True)
        with open("results/compression_comprehensive.json", "w") as f:
            json.dump(result, f, indent=2, default=str)

        logger.info("Comprehensive compression completed successfully")

        return {
            "message": "Comprehensive compression completed",
            "status": result.get("status"),
            "original_model": result.get("original_model"),
            "pruned_model": result.get("pruned_model"),
            "quantized_model": result.get("quantized_model"),
            "distilled_model": result.get("distilled_model"),
            "best_model": result.get("best_model"),
            "compression_summary": result.get("compression_summary", {}),
            "comparison_report": result.get("comparison_report", {})
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during comprehensive compression: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Comprehensive compression failed: {str(e)}"
        )