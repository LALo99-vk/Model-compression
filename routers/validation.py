"""
Validation Router - Phase 1: Dataset Validation & Conditioning
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import os
import logging
from services.dataset_validation_service import DatasetValidationService
from services.dataset_conditioning_service import DatasetConditioningService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()
validation_service = DatasetValidationService()
conditioning_service = DatasetConditioningService()


class ValidationRequest(BaseModel):
    dataset_path: str = Field(..., description="Path to the dataset file/directory")
    model_type: str = Field(..., description="Model type: 'decision_tree', 'cnn', or 'rnn'")


class ConditioningRequest(BaseModel):
    dataset_path: str = Field(..., description="Path to the dataset file/directory")
    model_type: str = Field(..., description="Model type: 'decision_tree', 'cnn', or 'rnn'")
    auto_fix: bool = Field(True, description="Whether to automatically fix issues")


@router.post("/validate")
async def validate_dataset(request: ValidationRequest):
    """
    Phase 1.1-1.2: Validate dataset against model requirements
    Returns comprehensive validation report - NEVER raises exceptions
    """
    try:
        # Validate inputs - return JSON error instead of raising
        if request.model_type.lower() not in ['decision_tree', 'cnn', 'rnn']:
            return {
                "status": "error",
                "message": f"Invalid model_type: {request.model_type}. Must be 'decision_tree', 'cnn', or 'rnn'",
                "issues": [],
                "can_autofix": False
            }
        
        if not os.path.exists(request.dataset_path):
            return {
                "status": "error",
                "message": f"Dataset not found: {request.dataset_path}",
                "issues": [f"File not found: {request.dataset_path}"],
                "can_autofix": False
            }
        
        # Perform validation
        validation_result = validation_service.validate_dataset(
            request.dataset_path,
            request.model_type
        )
        
        # Generate human-friendly report
        report_text = validation_service.generate_validation_report(validation_result)
        
        # Save validation report
        os.makedirs("results", exist_ok=True)
        with open("results/dataset_validation_report.json", "w") as f:
            import json
            json.dump(validation_result, f, indent=2, default=str)
        with open("results/dataset_validation_report.txt", "w") as f:
            f.write(report_text)
        
        # Return proper JSON response format
        if validation_result['is_valid']:
            return {
                "status": "valid",
                "message": "Dataset validation passed successfully.",
                "issues": [],
                "warnings": validation_result.get('warnings', []),
                "can_autofix": False,
                "info": validation_result.get('info', {}),
                "report_text": report_text
            }
        else:
            return {
                "status": "invalid",
                "message": "Dataset validation failed.",
                "issues": validation_result.get('issues', []),
                "warnings": validation_result.get('warnings', []),
                "can_autofix": validation_result.get('can_auto_fix', False),
                "fix_suggestions": validation_result.get('fix_suggestions', []),
                "info": validation_result.get('info', {}),
                "report_text": report_text
            }
        
    except Exception as e:
        logger.error(f"Error validating dataset: {str(e)}", exc_info=True)
        # NEVER raise exception - return JSON error
        return {
            "status": "error",
            "message": f"Validation error: {str(e)}",
            "issues": [str(e)],
            "can_autofix": False
        }


@router.post("/condition")
async def condition_dataset(request: ConditioningRequest):
    """
    Phase 2: Auto-fix dataset issues
    NEVER raises exceptions - returns JSON responses
    """
    try:
        # Validate inputs - return JSON error instead of raising
        if request.model_type.lower() not in ['decision_tree', 'cnn', 'rnn']:
            return {
                "status": "error",
                "message": f"Invalid model_type: {request.model_type}",
                "issues": [],
                "backup_path": None
            }
        
        if not os.path.exists(request.dataset_path):
            return {
                "status": "error",
                "message": f"Dataset not found: {request.dataset_path}",
                "issues": [],
                "backup_path": None
            }
        
        # First, validate the dataset
        validation_result = validation_service.validate_dataset(
            request.dataset_path,
            request.model_type
        )
        
        # If already valid, return early
        if validation_result['is_valid']:
            return {
                "status": "valid",
                "message": "Dataset is already valid, no conditioning needed",
                "new_path": request.dataset_path,
                "changes_made": [],
                "backup_path": None
            }
        
        # Perform conditioning
        conditioning_result = conditioning_service.condition_dataset(
            request.dataset_path,
            request.model_type,
            validation_result
        )
        
        if not conditioning_result['success']:
            return {
                "status": "error",
                "message": f"Conditioning failed: {conditioning_result.get('report', 'Unknown error')}",
                "issues": [],
                "backup_path": conditioning_result.get('backup_path')
            }
        
        # Re-validate after conditioning
        re_validation = validation_service.validate_dataset(
            conditioning_result['conditioned_path'],
            request.model_type
        )
        
        # Save conditioning report
        os.makedirs("results", exist_ok=True)
        conditioning_report = {
            "conditioning_result": conditioning_result,
            "re_validation": re_validation,
            "original_validation": validation_result
        }
        with open("results/dataset_conditioning_report.json", "w") as f:
            import json
            json.dump(conditioning_report, f, indent=2, default=str)
        
        if not re_validation['is_valid']:
            return {
                "status": "invalid_after_fix",
                "message": "Auto-fix attempted, but issues remain.",
                "issues": re_validation.get('issues', []),
                "backup_path": conditioning_result.get('backup_path'),
                "changes_made": conditioning_result.get('changes_made', [])
            }
        
        return {
            "status": "fixed",
            "message": "Dataset successfully fixed.",
            "new_path": conditioning_result['conditioned_path'],
            "backup_path": conditioning_result.get('backup_path'),
            "changes_made": conditioning_result.get('changes_made', []),
            "warnings": re_validation.get('warnings', [])
        }
        
    except Exception as e:
        logger.error(f"Error conditioning dataset: {str(e)}", exc_info=True)
        # NEVER raise exception - return JSON error
        return {
            "status": "error",
            "message": f"Conditioning failed: {str(e)}",
            "issues": [],
            "backup_path": None
        }


@router.get("/report")
async def get_validation_report():
    """Get the last validation report"""
    report_path = "results/dataset_validation_report.json"
    
    if not os.path.exists(report_path):
        raise HTTPException(
            status_code=404,
            detail="No validation report found. Please validate a dataset first."
        )
    
    import json
    with open(report_path, "r") as f:
        report = json.load(f)
    
    report_text = validation_service.generate_validation_report(report)
    
    return {
        "validation_result": report,
        "report": report_text
    }

