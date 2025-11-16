"""
Comparison Router - Compares original vs compressed model
"""

from fastapi import APIRouter, HTTPException
import json
import os
import time

router = APIRouter()


@router.get("/compare")
async def compare_models():
    """
    Compare original and compressed models
    Returns side-by-side metrics comparison
    """
    # Load metrics
    original_metrics_path = "results/original_metrics.json"
    compressed_metrics_path = "results/compressed_metrics.json"

    if not os.path.exists(original_metrics_path):
        raise HTTPException(
            status_code=404,
            detail="Original model metrics not found. Please train and evaluate first."
        )

    if not os.path.exists(compressed_metrics_path):
        raise HTTPException(
            status_code=404,
            detail="Compressed model metrics not found. Please compress and evaluate first."
        )

    with open(original_metrics_path, "r") as f:
        original_metrics = json.load(f)

    with open(compressed_metrics_path, "r") as f:
        compressed_metrics = json.load(f)

    # Get file sizes
    original_model_path = "models/original_model.pt"
    compressed_model_path = "models/compressed_model.pt"

    # Try different extensions
    for ext in ['.pt', '.h5', '.pkl']:
        if os.path.exists(original_model_path.replace('.pt', ext)):
            original_model_path = original_model_path.replace('.pt', ext)
            break

    for ext in ['.pt', '.h5', '.pkl']:
        if os.path.exists(compressed_model_path.replace('.pt', ext)):
            compressed_model_path = compressed_model_path.replace('.pt', ext)
            break

    original_size = os.path.getsize(original_model_path) if os.path.exists(original_model_path) else 0
    compressed_size = os.path.getsize(compressed_model_path) if os.path.exists(compressed_model_path) else 0

    # Calculate differences
    size_reduction = ((original_size - compressed_size) / original_size * 100) if original_size > 0 else 0

    accuracy_diff = compressed_metrics.get('accuracy', 0) - original_metrics.get('accuracy', 0)

    inference_time_diff = compressed_metrics.get('inference_time', 0) - original_metrics.get('inference_time', 0)
    speedup = (original_metrics.get('inference_time', 1) / compressed_metrics.get('inference_time',
                                                                                  1)) if compressed_metrics.get(
        'inference_time', 0) > 0 else 1

    comparison = {
        "file_size": {
            "original_mb": round(original_size / (1024 * 1024), 2),
            "compressed_mb": round(compressed_size / (1024 * 1024), 2),
            "reduction_percent": round(size_reduction, 2),
            "compression_ratio": round(original_size / compressed_size, 2) if compressed_size > 0 else 0
        },
        "accuracy": {
            "original": round(original_metrics.get('accuracy', 0), 4),
            "compressed": round(compressed_metrics.get('accuracy', 0), 4),
            "difference": round(accuracy_diff, 4),
            "difference_percent": round(accuracy_diff * 100, 2)
        },
        "inference_time": {
            "original_ms": round(original_metrics.get('inference_time', 0) * 1000, 2),
            "compressed_ms": round(compressed_metrics.get('inference_time', 0) * 1000, 2),
            "speedup": round(speedup, 2)
        },
        "detailed_metrics": {
            "original": original_metrics,
            "compressed": compressed_metrics
        }
    }

    # Save comparison
    with open("results/model_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    return comparison


@router.get("/summary")
async def get_comparison_summary():
    """Get a brief summary of model comparison"""
    comparison_path = "results/model_comparison.json"

    if not os.path.exists(comparison_path):
        # Generate comparison if not exists
        return await compare_models()

    with open(comparison_path, "r") as f:
        comparison = json.load(f)

    summary = {
        "size_reduction": f"{comparison['file_size']['reduction_percent']}%",
        "accuracy_change": f"{comparison['accuracy']['difference_percent']:+.2f}%",
        "speedup": f"{comparison['inference_time']['speedup']:.2f}x",
        "compression_ratio": f"{comparison['file_size']['compression_ratio']:.2f}:1"
    }

    return summary


@router.get("/table")
async def get_comparison_table():
    """Get comparison in table format"""
    comparison_path = "results/model_comparison.json"

    if not os.path.exists(comparison_path):
        return await compare_models()

    with open(comparison_path, "r") as f:
        comparison = json.load(f)

    table = {
        "headers": ["Metric", "Original Model", "Compressed Model", "Change"],
        "rows": [
            [
                "File Size (MB)",
                comparison['file_size']['original_mb'],
                comparison['file_size']['compressed_mb'],
                f"-{comparison['file_size']['reduction_percent']}%"
            ],
            [
                "Accuracy",
                f"{comparison['accuracy']['original']:.4f}",
                f"{comparison['accuracy']['compressed']:.4f}",
                f"{comparison['accuracy']['difference_percent']:+.2f}%"
            ],
            [
                "Inference Time (ms)",
                comparison['inference_time']['original_ms'],
                comparison['inference_time']['compressed_ms'],
                f"{comparison['inference_time']['speedup']:.2f}x faster"
            ],
            [
                "Precision",
                f"{comparison['detailed_metrics']['original'].get('precision', 0):.4f}",
                f"{comparison['detailed_metrics']['compressed'].get('precision', 0):.4f}",
                f"{(comparison['detailed_metrics']['compressed'].get('precision', 0) - comparison['detailed_metrics']['original'].get('precision', 0)) * 100:+.2f}%"
            ],
            [
                "Recall",
                f"{comparison['detailed_metrics']['original'].get('recall', 0):.4f}",
                f"{comparison['detailed_metrics']['compressed'].get('recall', 0):.4f}",
                f"{(comparison['detailed_metrics']['compressed'].get('recall', 0) - comparison['detailed_metrics']['original'].get('recall', 0)) * 100:+.2f}%"
            ],
            [
                "F1-Score",
                f"{comparison['detailed_metrics']['original'].get('f1_score', 0):.4f}",
                f"{comparison['detailed_metrics']['compressed'].get('f1_score', 0):.4f}",
                f"{(comparison['detailed_metrics']['compressed'].get('f1_score', 0) - comparison['detailed_metrics']['original'].get('f1_score', 0)) * 100:+.2f}%"
            ]
        ]
    }

    return table