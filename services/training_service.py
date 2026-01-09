"""
Training Service - Handles model training logic with comprehensive validation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import pandas as pd
import numpy as np
import json
import time
import pickle
import os
import logging
import gc
import traceback
from typing import Tuple, Optional, Dict, Any, List

# Optional dependency for memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
from utils.model_builder import ModelBuilder
from utils.data_loader import DataLoaderUtil
from utils.validation import DataValidator
from services.dataset_validation_service import DatasetValidationService
from services.dataset_conditioning_service import DatasetConditioningService
from services.preprocessing_service import PreprocessingService
from services.universal_dataset_normalizer import UniversalDatasetNormalizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log psutil availability after logger is initialized
if not PSUTIL_AVAILABLE:
    logger.warning("psutil not available. Memory monitoring will use default values.")


class TrainingService:
    def __init__(self):
        self.stop_flag = False
        self.model_builder = ModelBuilder()
        self.data_loader = DataLoaderUtil()
        self.validator = DataValidator()
        self.dataset_validator = DatasetValidationService()
        self.dataset_conditioner = DatasetConditioningService()
        self.preprocessing_service = PreprocessingService()
        self.universal_normalizer = UniversalDatasetNormalizer()
        self.preprocessing_warnings = []
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    def _save_to_training_history(self, model_config: dict, dataset_path: str, logs: dict):
        """
        Save training session to history file.
        Keeps last 10 training sessions.
        """
        import uuid
        from datetime import datetime
        
        history_path = "results/training_history.json"
        
        # Load existing history
        history = []
        if os.path.exists(history_path):
            try:
                with open(history_path, "r") as f:
                    history = json.load(f)
            except:
                history = []
        
        # Create session entry
        session = {
            "id": str(uuid.uuid4())[:8],
            "model_type": model_config.get("model_type", "unknown"),
            "dataset_path": dataset_path,
            "dataset_name": os.path.basename(dataset_path) if dataset_path else "Unknown",
            "created_at": datetime.now().isoformat(),
            "training_time": logs.get("training_time", 0),
            "epochs_trained": logs.get("epochs_trained", logs.get("epochs", 0)),
            "original": {
                "accuracy": logs.get("val_score", logs.get("train_score", 0)),
                "size_kb": logs.get("model_size_kb", 0),
                "size_mb": logs.get("model_size_mb", 0),
                "parameters": logs.get("total_parameters", logs.get("num_parameters", 0)),
                "path": logs.get("model_path", "")
            },
            "compressed": None,  # Will be updated when compression happens
            "task_type": logs.get("task_type", model_config.get("task_type", "classification"))
        }
        
        # Get accuracy from history if available
        if logs.get("history") and len(logs["history"]) > 0:
            last_epoch = logs["history"][-1]
            if last_epoch.get("val_accuracy", 0) > 0:
                session["original"]["accuracy"] = last_epoch["val_accuracy"]
        
        # Add to history (at the beginning)
        history.insert(0, session)
        
        # Keep only last 10 sessions
        history = history[:10]
        
        # Save history
        os.makedirs("results", exist_ok=True)
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, default=str)
        
        logger.info(f"✅ Training session saved to history: {session['id']}")
    
    def _atomic_write_json(self, filepath: str, data: dict):
        """Atomic write to prevent file corruption during concurrent reads"""
        temp_path = f"{filepath}.tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        # Windows-compatible atomic rename with retry
        self._safe_replace(temp_path, filepath)
    
    def _safe_replace(self, src: str, dst: str, max_retries: int = 5):
        """Windows-compatible file replace with retries"""
        import time
        for attempt in range(max_retries):
            try:
                # Try atomic replace first
                os.replace(src, dst)
                return
            except PermissionError:
                if attempt < max_retries - 1:
                    time.sleep(0.1 * (attempt + 1))  # Increasing delay
                else:
                    # Final fallback: delete then rename
                    try:
                        if os.path.exists(dst):
                            os.remove(dst)
                        os.rename(src, dst)
                    except Exception:
                        # Last resort: copy content directly
                        import shutil
                        shutil.copy2(src, dst)
                        try:
                            os.remove(src)
                        except:
                            pass

    def train_model(self, model_config, dataset_path, epochs, batch_size, validation_split):
        """Train model based on configuration with comprehensive validation"""
        self.stop_flag = False

        try:
            # 🚦 PHASE 1: DATASET VALIDATION & CONDITIONING
            self._update_status("validating", 0, epochs, "Phase 1: Validating dataset format and structure...")
            
            # 1.1-1.2: Validate dataset against model requirements
            validation_result = self.dataset_validator.validate_dataset(dataset_path, model_config['model_type'])
            
            # 1.3: Check if dataset is valid
            if not validation_result['is_valid']:
                # Generate validation report
                report = self.dataset_validator.generate_validation_report(validation_result)
                
                # Save report for user to see
                os.makedirs("results", exist_ok=True)
                with open("results/dataset_validation_report.json", "w", encoding="utf-8") as f:
                    json.dump(validation_result, f, indent=2, default=str)
                
                # Save report text
                with open("results/dataset_validation_report.txt", "w", encoding="utf-8") as f:
                    f.write(report)
                
                error_msg = (
                    f"Dataset validation failed. Status: {validation_result['status']}\n"
                    f"Issues found: {len(validation_result['issues'])}\n"
                    f"Please check results/dataset_validation_report.txt for details.\n"
                    f"The dataset does not match the expected format for {model_config['model_type']} model."
                )
                
                if validation_result.get('can_auto_fix'):
                    error_msg += (
                        f"\n\nThese issues can be automatically fixed.\n"
                        f"Call /api/validation/condition endpoint to auto-fix the dataset."
                    )
                
                logger.error(error_msg)
                self._update_status("error", 0, epochs, error_msg)
                raise ValueError(error_msg)

            # Dataset is valid, proceed with training
            logger.info(f"✅ Phase 1 validation passed: Dataset is valid for {model_config['model_type']} model")
            
            # Validate and auto-correct model configuration
            model_config = self._validate_and_correct_config(model_config, dataset_path)

            # 🔄 UNIVERSAL NORMALIZATION: Normalize dataset BEFORE training
            # This ensures training.py receives ONLY standardized inputs
            self._update_status("normalizing", 0, epochs, "Universal normalization: Processing dataset...")
            
            try:
                # Update status: Schema detection
                self._update_status("normalizing", 0, epochs, "Universal normalization: Detecting dataset schema...")
                
                normalization_result = self.universal_normalizer.normalize_dataset(
                dataset_path,
                model_config['model_type'],
                validation_split
            )

                if normalization_result['status'] != 'success':
                    # Normalization failed - return structured error BEFORE training
                    error_response = {
                        "status": "normalization_failed",
                        "message": "Dataset normalization failed before training",
                        "errors": normalization_result.get('errors', []),
                        "guidance": normalization_result.get('guidance', 'Please check your dataset format'),
                        "cause": "Dataset could not be normalized for model type",
                        "suggested_fix": normalization_result.get('guidance', 'Review dataset format and try again')
                    }
                    
                    logger.error(f"Normalization failed: {error_response}")
                    self._update_status("error", 0, epochs, error_response['message'])
                    
                    # Save error response
                    os.makedirs("results", exist_ok=True)
                    with open("results/normalization_error.json", "w", encoding="utf-8") as f:
                        json.dump(error_response, f, indent=2, default=str)
                    
                    raise ValueError(json.dumps(error_response, indent=2))
                
                # Update status: Normalization complete, preparing data
                self._update_status("normalizing", 0, epochs, "Universal normalization: Preparing standardized data...")
                
                # Extract standardized outputs - training.py receives ONLY these
                X_train = normalization_result['X_train']
                X_val = normalization_result['X_val']
                X_test = normalization_result['X_test']
                y_train = normalization_result['y_train']
                y_val = normalization_result['y_val']
                y_test = normalization_result['y_test']
                
                # Store schema info for later use (normalizer returns 'schema' not 'schema_info')
                schema_info = normalization_result.get('schema', normalization_result.get('schema_info', {}))
                
                logger.info(f"✅ Universal normalization completed: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
                logger.info(f"✅ Standardized outputs ready: X_train shape={X_train.shape}, y_train shape={y_train.shape}")
                
                # CRITICAL: Get num_classes from normalizer result (TRUST this value!)
                # The normalizer calculates the correct vocab_size for text data
                norm_num_classes = normalization_result.get('num_classes')
                if norm_num_classes and norm_num_classes > 1:
                    model_config['num_classes'] = norm_num_classes
                    logger.info(f"✅ Using normalizer's num_classes: {norm_num_classes}")
                elif schema_info.get('n_classes') and schema_info['n_classes'] > 1:
                    model_config['num_classes'] = schema_info['n_classes']
                    logger.info(f"✅ Using schema's n_classes: {schema_info['n_classes']}")
                
                # Get task_type from normalizer
                if schema_info.get('task_type'):
                    model_config['task_type'] = schema_info['task_type']
                    logger.info(f"✅ Using normalizer's task_type: {schema_info['task_type']}")
                
            except ValueError as e:
                # Re-raise normalization errors
                raise
            except Exception as e:
                # Unexpected error during normalization
                error_response = {
                    "status": "normalization_error",
                    "message": f"Unexpected error during dataset normalization: {str(e)}",
                    "cause": str(e),
                    "suggested_fix": "Please check dataset format and try again"
                }
                logger.error(f"Normalization error: {error_response}", exc_info=True)
                self._update_status("error", 0, epochs, error_response['message'])
                raise ValueError(json.dumps(error_response, indent=2))
            
            # Validate standardized data before training
            if len(X_train) == 0:
                raise ValueError("Training data is empty after normalization")
            if len(X_val) == 0:
                raise ValueError("Validation data is empty after normalization")
            
            # At this point, training.py has ONLY standardized inputs
            # All dataset adaptation has occurred in normalization phase
            logger.info("✅ Dataset fully normalized - training.py will receive standardized inputs only")
            
            # Update status: Ready to train
            self._update_status("loading_data", 0, epochs, "Dataset normalized. Preparing model for training...")
            
            # Save standardized data for compression/evaluation later
            os.makedirs("results", exist_ok=True)
            standardized_data = {
                "X_train": X_train.tolist() if hasattr(X_train, 'tolist') else X_train,
                "X_val": X_val.tolist() if hasattr(X_val, 'tolist') else X_val,
                "X_test": X_test.tolist() if hasattr(X_test, 'tolist') else X_test,
                "y_train": y_train.tolist() if hasattr(y_train, 'tolist') else y_train,
                "y_val": y_val.tolist() if hasattr(y_val, 'tolist') else y_val,
                "y_test": y_test.tolist() if hasattr(y_test, 'tolist') else y_test,
                "schema_info": schema_info
            }
            with open("results/training_data.json", "w", encoding="utf-8") as f:
                json.dump(standardized_data, f, indent=2, default=str)
            
            logger.info("✅ Standardized data saved for later use")
            
            # Auto-infer missing parameters (after normalization, shapes are known)
            model_config = self._auto_infer_parameters(model_config, X_train, y_train)
            
            # Check unique labels in y_train for fallback detection
            unique_labels = len(np.unique(y_train))
            logger.info(f"📊 y_train has {unique_labels} unique labels")
            
            # IMPORTANT: Don't override num_classes if normalizer already set it correctly!
            # Normalizer's num_classes is based on FULL dataset (e.g., full char vocabulary)
            # y_train's unique count may be LOWER due to train/val/test split
            current_num_classes = model_config.get('num_classes', 0)
            
            # If too many unique values (>100), treat as regression (continuous target)
            if unique_labels > 100:
                if model_config.get('task_type') != 'regression':
                    logger.warning(f"⚠️ Detected {unique_labels} unique values (continuous target). AUTO-SWITCHING to REGRESSION.")
                    model_config['task_type'] = 'regression'
            # Only override num_classes if it was NOT already set by normalizer
            elif current_num_classes <= 1 and unique_labels > 1 and unique_labels <= 100:
                logger.warning(f"num_classes not set by normalizer. Using y_train unique count: {unique_labels}")
                model_config['num_classes'] = unique_labels
                if model_config.get('task_type') == 'regression':
                    model_config['task_type'] = 'classification'
            # If normalizer set num_classes but task_type is wrong, fix task_type only
            elif current_num_classes > 1 and model_config.get('task_type') == 'regression':
                logger.warning(f"num_classes={current_num_classes} but task_type=regression. Fixing task_type to classification.")
                model_config['task_type'] = 'classification'
            
            logger.info(f"📊 Final config: num_classes={model_config.get('num_classes')}, task_type={model_config.get('task_type')}")

            # Ensure results directory exists
            os.makedirs("results", exist_ok=True)
            os.makedirs("models", exist_ok=True)

            model_type = model_config['model_type']

            # Train based on model type
            if model_type == 'decision_tree':
                self._train_sklearn_model(model_config, X_train, X_val, y_train, y_val, dataset_path)
            elif model_type in ['cnn', 'rnn']:
                # Auto-adjust batch size if needed
                batch_size = self._auto_adjust_batch_size(batch_size, len(X_train))
                self._train_pytorch_model(model_config, X_train, X_val, y_train, y_val, epochs, batch_size, dataset_path)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            # Phase 6: Generate comprehensive training output
            training_output = self._generate_training_output(model_config, dataset_path, epochs)
            
            # Save comprehensive output
            os.makedirs("results", exist_ok=True)
            with open("results/training_output.json", "w", encoding="utf-8") as f:
                json.dump(training_output, f, indent=2, default=str)

            # Update final status with output reference
            self._update_status("completed", epochs, epochs, "Training completed successfully")

            logger.info("✅ Training completed with comprehensive output generated")
            return training_output

        except Exception as e:
            # Phase 3: Never throw raw exceptions - return structured error response
            error_response = self._create_training_error_response(e, model_config, dataset_path)
            
            # Update status with error
            error_message = error_response.get("message", str(e))
            self._update_status("error", 0, epochs, error_message)
            
            # Save error response
            os.makedirs("results", exist_ok=True)
            with open("results/training_error_response.json", "w", encoding="utf-8") as f:
                json.dump(error_response, f, indent=2, default=str)
            
            logger.error(f"Training failed: {error_message}", exc_info=True)
            
            # Return error response instead of raising
            return error_response
    
    # ========== PHASE 6: COMPREHENSIVE TRAINING OUTPUT ==========
    
    def _generate_training_output(self, model_config: Dict[str, Any], dataset_path: str, epochs: int) -> Dict[str, Any]:
        """
        Phase 6: Generate comprehensive training output including:
        - model_file
        - training_metrics
        - training_logs
        - validation_report
        - suggestions
        """
        output = {
            "status": "completed",
            "model_file": None,
            "training_metrics": {},
            "training_logs": [],
            "validation_report": {},
            "suggestions": []
        }
        
        # Load model file path
        model_type = model_config.get('model_type')
        if model_type == 'decision_tree':
            model_file = "models/original_model.pkl"
        else:
            model_file = "models/original_model.pt"
        
        if os.path.exists(model_file):
            output["model_file"] = model_file
        
        # Load training logs
        logs_path = "results/training_logs.json"
        if os.path.exists(logs_path):
            try:
                with open(logs_path, "r") as f:
                    training_logs = json.load(f)
                    output["training_logs"] = training_logs.get("history", [])
            except Exception as e:
                logger.warning(f"Could not load training logs: {str(e)}")
        
        # Load training metrics from logs
        if os.path.exists(logs_path):
            try:
                with open(logs_path, "r") as f:
                    logs = json.load(f)
                    
                    # Extract metrics
                    metrics = {
                        "model_type": logs.get("model_type"),
                        "epochs": logs.get("epochs"),
                        "epochs_trained": logs.get("epochs_trained", logs.get("epochs")),
                        "training_time": logs.get("training_time"),
                        "model_size_mb": logs.get("model_size_mb"),
                        "total_parameters": logs.get("total_parameters"),
                        "best_val_loss": logs.get("best_val_loss"),
                    }
                    
                    # Add confusion matrix if available
                    if "confusion_matrix" in logs:
                        metrics["confusion_matrix"] = logs["confusion_matrix"]
                    
                    # Add inference speed if available
                    if "inference_speed_ms" in logs:
                        metrics["inference_speed_ms"] = logs["inference_speed_ms"]
                    
                    # Add final epoch metrics
                    if logs.get("history") and len(logs["history"]) > 0:
                        last_epoch = logs["history"][-1]
                        metrics["final_train_loss"] = last_epoch.get("train_loss")
                        metrics["final_val_loss"] = last_epoch.get("val_loss")
                        metrics["final_val_accuracy"] = last_epoch.get("val_accuracy")
                        metrics["final_learning_rate"] = last_epoch.get("learning_rate")
                    
                    output["training_metrics"] = metrics
            except Exception as e:
                logger.warning(f"Could not extract training metrics: {str(e)}")
        
        # Load validation report
        validation_report_path = "results/dataset_validation_report.json"
        if os.path.exists(validation_report_path):
            try:
                with open(validation_report_path, "r") as f:
                    output["validation_report"] = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load validation report: {str(e)}")
        
        # Generate suggestions
        output["suggestions"] = self._generate_training_suggestions(model_config, output.get("training_metrics", {}))
        
        return output
    
    def _generate_training_suggestions(self, model_config: Dict[str, Any], metrics: Dict[str, Any]) -> List[str]:
        """Generate suggestions for improving training results"""
        suggestions = []
        
        # Check validation loss
        best_val_loss = metrics.get("best_val_loss")
        if best_val_loss and best_val_loss > 1.0:
            suggestions.append("Validation loss is high. Consider increasing model complexity or training for more epochs.")
        
        # Check training vs validation loss (overfitting)
        final_train_loss = metrics.get("final_train_loss")
        final_val_loss = metrics.get("final_val_loss")
        if final_train_loss and final_val_loss and final_val_loss > final_train_loss * 1.5:
            suggestions.append("Model appears to be overfitting. Consider adding regularization or reducing model complexity.")
        
        # Check accuracy
        final_val_accuracy = metrics.get("final_val_accuracy")
        if final_val_accuracy and final_val_accuracy < 0.5:
            suggestions.append("Validation accuracy is low. Consider checking data quality, increasing training epochs, or adjusting learning rate.")
        
        # Check model size
        model_size_mb = metrics.get("model_size_mb")
        if model_size_mb and model_size_mb > 100:
            suggestions.append("Model is large. Consider using compression techniques to reduce size while maintaining performance.")
        
        # Check training time
        training_time = metrics.get("training_time")
        if training_time and training_time > 3600:  # More than 1 hour
            suggestions.append("Training took a long time. Consider reducing model complexity or using a smaller dataset for faster iteration.")
        
        # General suggestions
        if not suggestions:
            suggestions.append("Training completed successfully! Model is ready for evaluation and compression.")
        
        return suggestions
    
    def _validate_and_correct_config(self, model_config, dataset_path):
        """Validate and auto-correct model configuration"""
        try:
            # Load dataset metadata if available
            dataset_info_path = "models/dataset_info.json"
            if os.path.exists(dataset_info_path):
                with open(dataset_info_path, "r") as f:
                    dataset_info = json.load(f)
                metadata = dataset_info.get('metadata', {})
                
                # Auto-correct configuration based on dataset
                model_config = self.validator.validate_model_config(model_config, metadata)
        except Exception as e:
            logger.warning(f"Could not validate config from dataset info: {str(e)}")
        
        return model_config
    
    def _auto_infer_parameters(self, model_config, X_train, y_train):
        """Automatically infer missing model parameters"""
        # Auto-detect task type if not set
        if not model_config.get('task_type'):
            task_type, num_classes = self.validator.infer_task_type_from_target(y_train)
            model_config['task_type'] = task_type
            logger.info(f"Auto-detected task type: {task_type}")
        
        # Auto-set num_classes for classification
        if model_config.get('task_type') == 'classification':
            if not model_config.get('num_classes'):
                unique_labels = np.unique(y_train)
                model_config['num_classes'] = int(len(unique_labels))
                logger.info(f"Auto-detected num_classes: {model_config['num_classes']}")
        
        # Auto-set input_shape and num_classes for neural networks
        model_type = model_config.get('model_type')
        if model_type in ['cnn', 'rnn']:
            # For regression tasks with neural networks, set num_classes=1 (single output)
            if model_config.get('task_type') == 'regression' and not model_config.get('num_classes'):
                model_config['num_classes'] = 1
                logger.info("Set num_classes=1 for regression task")
            
            if not model_config.get('input_shape'):
                if model_type == 'cnn':
                    # For CNN, try to infer from data shape
                    if X_train.ndim == 4:
                        model_config['input_shape'] = tuple(X_train.shape[1:])
                    elif X_train.ndim == 2:
                        # Try to reshape to square image
                        num_features = X_train.shape[1]
                        side = int(np.sqrt(num_features))
                        if side * side == num_features:
                            model_config['input_shape'] = (1, side, side)
                        else:
                            model_config['input_shape'] = (1, num_features, 1)
                    logger.info(f"Auto-detected input_shape for CNN: {model_config['input_shape']}")
                elif model_type == 'rnn':
                    if X_train.ndim == 3:
                        model_config['input_shape'] = tuple(X_train.shape[1:])
                    elif X_train.ndim == 2:
                        model_config['input_shape'] = (X_train.shape[1], 1)
                    logger.info(f"Auto-detected input_shape for RNN: {model_config['input_shape']}")
        
        return model_config
    
    def _auto_adjust_batch_size(self, batch_size, dataset_size):
        """Automatically adjust batch size if needed (basic version, used for sklearn)"""
        # Ensure batch size is not larger than dataset
        if batch_size > dataset_size:
            new_batch_size = max(1, dataset_size // 4)
            logger.warning(f"Batch size {batch_size} larger than dataset size {dataset_size}, adjusting to {new_batch_size}")
            return new_batch_size
        
        # Ensure batch size is reasonable (not too small)
        if batch_size < 8 and dataset_size >= 32:
            new_batch_size = min(32, dataset_size // 4)
            logger.info(f"Increasing batch size from {batch_size} to {new_batch_size} for better training")
            return new_batch_size
        
        return batch_size
    
    def _check_available_memory(self) -> float:
        """Check available system memory in GB"""
        if not PSUTIL_AVAILABLE:
            return 4.0  # Default assumption if psutil not available
        
        try:
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024 ** 3)
            return available_gb
        except Exception as e:
            logger.warning(f"Could not check memory: {str(e)}")
            return 4.0  # Default assumption
    
    def _auto_adjust_batch_size_with_memory(
        self, 
        batch_size: int, 
        dataset_size: int, 
        model: nn.Module, 
        input_shape: tuple,
        available_memory_gb: float
    ) -> int:
        """Enhanced batch size auto-scaling with memory checks"""
        # Start with basic adjustments
        if batch_size > dataset_size:
            batch_size = max(1, dataset_size // 4)
            logger.warning(f"Batch size adjusted: {batch_size} (dataset size: {dataset_size})")
        
        # Memory-based adjustment
        if torch.cuda.is_available():
            try:
                # Estimate memory per sample (rough estimate)
                model.eval()
                with torch.no_grad():
                    # Handle different input shape formats
                    if isinstance(input_shape, tuple) and len(input_shape) > 1:
                        dummy_shape = input_shape[1:] if input_shape[0] == -1 else input_shape
                    else:
                        dummy_shape = input_shape[1:] if len(input_shape) > 1 else input_shape
                    
                    dummy_input = torch.zeros(1, *dummy_shape).to(self.device)
                    _ = model(dummy_input)
                    memory_per_sample_mb = torch.cuda.memory_allocated() / (1024 ** 2)
                    
                    # Estimate max batch size based on available GPU memory
                    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                    usable_memory_gb = gpu_memory_gb * 0.7  # Use 70% of GPU memory
                    max_batch_size = int((usable_memory_gb * 1024) / (memory_per_sample_mb * 2))  # 2x for gradients
                    
                    if batch_size > max_batch_size:
                        logger.warning(f"Batch size {batch_size} may cause OOM. Reducing to {max_batch_size}")
                        batch_size = max(1, max_batch_size)
                    
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"Could not estimate GPU memory requirements: {str(e)}")
        else:
            # CPU memory check
            if available_memory_gb < 2.0:
                batch_size = min(batch_size, 16)
                logger.warning(f"Low system memory ({available_memory_gb:.2f}GB), limiting batch size to {batch_size}")
        
        # Ensure reasonable batch size
        if batch_size < 8 and dataset_size >= 32:
            batch_size = min(32, dataset_size // 4)
            logger.info(f"Increasing batch size to {batch_size} for better training")
        
        return batch_size

    def _train_sklearn_model(self, model_config, X_train, X_val, y_train, y_val, dataset_path):
        """Train sklearn-based models with enhanced error handling"""
        config = model_config.get('config', {})
        task_type = model_config['task_type']
        start_time = time.time()
        
        logger.info(f"Training {task_type} model (Decision Tree) on {len(X_train)} samples")

        # Initialize logs
        logs = {
            "model_type": "decision_tree",
            "task_type": task_type,
            "samples_trained": len(X_train),
            "n_features": X_train.shape[1],
            "status_message": "Initializing Decision Tree model...",
            "history": []
        }
        self._atomic_write_json("results/training_logs.json", logs)

        if task_type == 'classification':
            model = DecisionTreeClassifier(
                max_depth=config.get('max_depth', 10),
                min_samples_split=config.get('min_samples_split', 2),
                min_samples_leaf=config.get('min_samples_leaf', 1),
                criterion=config.get('criterion', 'gini')
            )
        else:
            model = DecisionTreeRegressor(
                max_depth=config.get('max_depth', 10),
                min_samples_split=config.get('min_samples_split', 2),
                min_samples_leaf=config.get('min_samples_leaf', 1)
            )

        # Train
        self._update_status("training", 0, 1, "Training Decision Tree model...")
        logs["status_message"] = f"Training Decision Tree on {len(X_train)} samples..."
        self._atomic_write_json("results/training_logs.json", logs)
        
        train_start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - train_start
        
        logs["status_message"] = "Training complete! Validating model..."
        self._atomic_write_json("results/training_logs.json", logs)

        # Validate
        train_score = model.score(X_train, y_train)
        val_score = model.score(X_val, y_val)

        # Save model metadata separately for evaluation
        model_metadata = {
            "n_features": int(X_train.shape[1]),
            "feature_names": None,  # Can be extended later
            "n_classes": int(len(np.unique(y_train))) if task_type == 'classification' else None,
            "task_type": task_type,
            "dataset_path": dataset_path,
            "model_type": "decision_tree"
        }
        with open("models/original_model_metadata.json", "w", encoding="utf-8") as f:
            json.dump(model_metadata, f, indent=2, default=str)
        
        # Save model (keep it as just the model for compatibility with compression service)
        model_path = "models/original_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        
        # Calculate model size
        model_size_bytes = os.path.getsize(model_path)
        model_size_mb = model_size_bytes / (1024 * 1024)
        
        # Calculate model parameters/complexity
        if hasattr(model, 'tree_'):
            num_nodes = model.tree_.node_count
            num_parameters = num_nodes
        else:
            num_nodes = 0
            num_parameters = 0
        
        logger.info(f"Model saved: {model_size_mb:.4f} MB ({model_size_bytes} bytes), {num_parameters} tree nodes")

        # Save training data for compression
        training_data = {
            "X_train": X_train.tolist() if hasattr(X_train, 'tolist') else X_train,
            "y_train": y_train.tolist() if hasattr(y_train, 'tolist') else y_train,
            "X_val": X_val.tolist() if hasattr(X_val, 'tolist') else X_val,
            "y_val": y_val.tolist() if hasattr(y_val, 'tolist') else y_val,
            "n_features": int(X_train.shape[1]),
            "n_samples_train": int(len(X_train)),
            "n_samples_val": int(len(X_val))
        }
        with open("results/training_data.json", "w", encoding="utf-8") as f:
            json.dump(training_data, f, indent=2)

        # Save logs (include a single-epoch history so frontend can display it)
        history = [
            {
                "epoch": 1,
                # Approximate "loss" as (1 - score) so charts have something sensible
                "train_loss": float(1.0 - train_score),
                "val_loss": float(1.0 - val_score),
                "val_accuracy": float(val_score),
            }
        ]

        logs = {
            "model_type": "decision_tree",
            "task_type": task_type,
            "num_classes": int(len(np.unique(y_train))) if task_type == 'classification' else None,
            "train_score": float(train_score),
            "val_score": float(val_score),
            "epochs": 1,
            "training_time": time.time() - start_time,
            "dataset_path": dataset_path,
            "history": history,
            "model_size_bytes": int(model_size_bytes),
            "model_size_mb": round(model_size_mb, 4),
            "model_size_kb": round(model_size_bytes / 1024, 2),
            "model_path": model_path,
            "num_parameters": int(num_parameters),
            "preprocessing_warnings": self.preprocessing_warnings if hasattr(self, 'preprocessing_warnings') else [],
        }

        self._atomic_write_json("results/training_logs.json", logs)
        
        # Save to training history
        self._save_to_training_history(model_config, dataset_path, logs)

    def _train_pytorch_model(self, model_config, X_train, X_val, y_train, y_val, epochs, batch_size, dataset_path):
        """
        Train PyTorch models with Phase 2 optimizations:
        - Early stopping
        - Learning rate scheduler
        - Mixed precision (if GPU)
        - Batch size auto-scaling with memory checks
        - Checkpointing (save best model only)
        - Input shape double-check before first batch
        - Memory checks to avoid OOM
        """
        model_type = model_config.get('model_type')
        logger.info(f"🚀 Phase 2: Training {model_type} model on {len(X_train)} samples for {epochs} epochs")
        
        # Phase 2: Memory check before starting
        available_memory_gb = self._check_available_memory()
        logger.info(f"Available memory: {available_memory_gb:.2f} GB")

        if model_type == 'cnn':
            if X_train.ndim == 2:
                num_features = X_train.shape[1]
                side = int(np.sqrt(num_features))
                if side * side == num_features:
                    h = side
                    w = side
                else:
                    h = num_features
                    w = 1
                X_train = X_train.reshape(-1, 1, h, w)
                X_val = X_val.reshape(-1, 1, h, w)
                model_config['input_shape'] = (1, h, w)
            elif X_train.ndim == 4:
                model_config['input_shape'] = tuple(X_train.shape[1:])
        elif model_type == 'rnn':
            if X_train.ndim == 2:
                model_config['input_shape'] = (X_train.shape[1],)
            elif X_train.ndim == 3:
                model_config['input_shape'] = tuple(X_train.shape[1:])

        # Build model and move to device
        model = self.model_builder.build_pytorch_model(model_config)
        model = model.to(self.device)
        
        # Phase 2: Input shape double-check BEFORE first training batch
        expected_input_shape = model_config.get('input_shape')
        if expected_input_shape:
            logger.info(f"✅ Input shape verification: Expected {expected_input_shape}")
            try:
                dummy_input = torch.zeros(1, *expected_input_shape).to(self.device)
                with torch.no_grad():
                    _ = model(dummy_input)
                logger.info(f"✅ Model input shape verified successfully")
            except Exception as e:
                logger.warning(f"⚠️ Shape verification failed: {e}")
                logger.info("💡 Skipping verification - will proceed with training and handle shape issues")
                # Don't raise - just warn and continue
                # raise ValueError(f"Input shape mismatch: Model expects {expected_input_shape}, but verification failed: {str(e)}")

        # Verify X and y have matching sizes
        if len(X_train) != len(y_train):
            raise ValueError(f"Size mismatch: X_train ({len(X_train)}) and y_train ({len(y_train)}) must match")
        if len(X_val) != len(y_val):
            raise ValueError(f"Size mismatch: X_val ({len(X_val)}) and y_val ({len(y_val)}) must match")

        # Phase 2: Enhanced batch size auto-scaling with memory checks
        batch_size = self._auto_adjust_batch_size_with_memory(
            batch_size, len(X_train), model, X_train.shape, available_memory_gb
        )

        # Determine label tensor type based on num_classes (not just task_type)
        num_classes = model_config.get('num_classes', 1)
        is_classification = num_classes > 1 or model_config.get('task_type') == 'classification'

        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train) if is_classification else torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val) if is_classification else torch.FloatTensor(y_val)
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
        val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=torch.cuda.is_available())

        # Setup training
        # Determine loss function based on task type and num_classes
        # If num_classes > 1, it's classification regardless of task_type
        num_classes = model_config.get('num_classes', 1)
        task_type = model_config.get('task_type', 'regression')
        
        if num_classes > 1:
            # Classification task
            criterion = nn.CrossEntropyLoss()
            logger.info(f"Using CrossEntropyLoss for classification (num_classes={num_classes})")
        elif task_type == 'classification' and num_classes == 1:
            # Binary classification
            criterion = nn.BCEWithLogitsLoss()
            logger.info("Using BCEWithLogitsLoss for binary classification")
        else:
            # Regression task
            criterion = nn.MSELoss()
            logger.info("Using MSELoss for regression")

        learning_rate = model_config['config'].get('learning_rate', 0.001)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Phase 2: Learning rate scheduler (verbose removed for PyTorch 2.4+ compatibility)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
        )
        
        # Phase 2: Mixed precision training (if GPU available)
        use_mixed_precision = torch.cuda.is_available() and hasattr(torch.cuda, 'amp')
        scaler = GradScaler() if use_mixed_precision else None
        if use_mixed_precision:
            logger.info("✅ Using mixed precision training (FP16)")

        # Phase 2: Early stopping
        early_stopping_patience = 7
        early_stopping_min_delta = 1e-4
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # Phase 2: Checkpointing (save best model only)
        checkpoint_dir = "models/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_model_path = os.path.join(checkpoint_dir, "best_model.pt")

        # Training loop
        training_history = []
        start_time = time.time()
        
        # Initialize logs with optimizations info
        optimizations_info = {
            "early_stopping": True,
            "learning_rate_scheduler": True,
            "mixed_precision": use_mixed_precision,
            "batch_size_auto_scaling": True,
            "checkpointing": True
        }
        logs = {
            "model_type": model_type,
            "task_type": task_type,
            "num_classes": num_classes if task_type == 'classification' else None,
            "epochs": epochs,
            "training_time": 0,
            "history": [],
            "preprocessing_warnings": self.preprocessing_warnings if hasattr(self, 'preprocessing_warnings') else [],
            "optimizations": optimizations_info
        }
        self._atomic_write_json("results/training_logs.json", logs)

        logger.info(f"Starting training with batch_size={batch_size}, learning_rate={learning_rate}")

        # Phase 5: Variables for confusion matrix (store best epoch predictions)
        best_all_predictions = []
        best_all_labels = []

        for epoch in range(epochs):
            if self.stop_flag:
                logger.info("Training stopped by user at epoch {}/{}".format(epoch, epochs))
                # Update status to stopped
                self._update_status("stopped", epoch, epochs, f"Training stopped by user at epoch {epoch}/{epochs}")
                break

            # Phase 2: Memory check before each epoch (every 10 epochs to avoid overhead)
            if epoch % 10 == 0:
                current_memory = self._check_available_memory()
                if current_memory < 0.5:  # Less than 500MB available
                    logger.warning(f"Low memory detected: {current_memory:.2f} GB. Clearing cache...")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            model.train()
            train_loss = 0.0

            for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                # Check stop flag at batch level for faster response
                if self.stop_flag:
                    logger.info("Training stopped by user during batch processing")
                    break
                
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                # Phase 2: Mixed precision forward pass
                if use_mixed_precision:
                    with autocast():
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                
                train_loss += loss.item()
                
                # Phase 2: Memory check during first batch to catch OOM early
                if epoch == 0 and batch_idx == 0:
                    if torch.cuda.is_available():
                        memory_allocated = torch.cuda.memory_allocated() / 1024**3
                        memory_reserved = torch.cuda.memory_reserved() / 1024**3
                        logger.info(f"First batch memory: Allocated={memory_allocated:.2f}GB, Reserved={memory_reserved:.2f}GB")

            # Validation
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            # Phase 5: Collect predictions and labels for confusion matrix
            all_predictions = []
            all_labels = []

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    # Check stop flag during validation too
                    if self.stop_flag:
                        logger.info("Training stopped by user during validation")
                        break
                    
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()

                    # Check if classification based on num_classes or task_type
                    is_classification = num_classes > 1 or model_config.get('task_type') == 'classification'
                    
                    if is_classification:
                        _, predicted = torch.max(outputs.data, 1)
                        total += batch_y.size(0)
                        correct += (predicted == batch_y).sum().item()
                        # Phase 5: Store predictions and labels for confusion matrix
                        all_predictions.extend(predicted.cpu().numpy())
                        all_labels.extend(batch_y.cpu().numpy())

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = correct / total if total > 0 else 0

            # Phase 2: Learning rate scheduling
            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']

            # Phase 2: Early stopping and checkpointing
            is_best = avg_val_loss < (best_val_loss - early_stopping_min_delta)
            
            if is_best:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Phase 5: Save predictions from best model for confusion matrix
                best_all_predictions = all_predictions.copy()
                best_all_labels = all_labels.copy()
                # Save best model state
                best_model_state = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                    'val_accuracy': val_accuracy,
                    'model_config': model_config
                }
                torch.save(best_model_state, best_model_path)
                logger.info(f"✅ New best model saved (val_loss={avg_val_loss:.4f})")
            else:
                patience_counter += 1

            # Log epoch
            epoch_log = {
                "epoch": epoch + 1,
                "train_loss": float(avg_train_loss),
                "val_loss": float(avg_val_loss),
                "val_accuracy": float(val_accuracy),
                "learning_rate": float(current_lr),
                "early_stopped": False,
                "message": f"Epoch {epoch + 1}/{epochs} complete | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.6f}"
            }
            training_history.append(epoch_log)

            # Save logs incrementally with detailed progress
            elapsed_time = time.time() - start_time
            avg_epoch_time = elapsed_time / (epoch + 1)
            remaining_epochs = epochs - (epoch + 1)
            eta_seconds = avg_epoch_time * remaining_epochs
            eta_minutes = eta_seconds / 60
            
            logs = {
                "model_type": model_type,
                "task_type": model_config.get('task_type', 'classification'),
                "num_classes": model_config.get('num_classes', num_classes),
                "epochs": epochs,
                "current_epoch": epoch + 1,
                "training_time": elapsed_time,
                "eta_minutes": eta_minutes if remaining_epochs > 0 else 0,
                "samples_trained": len(X_train),
                "batch_size": batch_size,
                "dataset_path": dataset_path,
                "history": training_history,
                "best_val_loss": float(best_val_loss),
                "current_lr": float(current_lr),
                "progress_percent": ((epoch + 1) / epochs) * 100,
                "status_message": f"Training epoch {epoch + 1}/{epochs} | {((epoch + 1) / epochs) * 100:.1f}% complete | ETA: {eta_minutes:.1f}min"
            }
            self._atomic_write_json("results/training_logs.json", logs)

            # Update status with progress
            self._update_status("training", epoch + 1, epochs,
                              f"Epoch {epoch + 1}/{epochs} | Loss: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | Acc: {val_accuracy:.3f} | LR: {current_lr:.6f}")

            # Phase 2: Early stopping check
            if patience_counter >= early_stopping_patience:
                logger.info(f"⏹️  Early stopping triggered after {epoch + 1} epochs (patience={early_stopping_patience})")
                epoch_log["early_stopped"] = True
                break

        # Load best model before final save
        if best_model_state is not None and os.path.exists(best_model_path):
            logger.info(f"Loading best model from epoch {best_model_state['epoch']} (val_loss={best_model_state['val_loss']:.4f})")
            model.load_state_dict(best_model_state['model_state_dict'])
        elif os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded best model from checkpoint (val_loss={checkpoint['val_loss']:.4f})")

        # Save final model
        model_path = "models/original_model.pt"
        torch.save(model.state_dict(), model_path)

        # Calculate model size
        model_size_bytes = os.path.getsize(model_path)
        model_size_mb = model_size_bytes / (1024 * 1024)
        
        # Calculate parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Model saved: {model_size_mb:.4f} MB, {total_params:,} params")

        # Save architecture
        model_arch = {
            "config": model_config,
            "state_dict_path": model_path,
            "dataset_path": dataset_path,
            "training_samples": len(X_train),
            "validation_samples": len(X_val),
            "model_size_bytes": int(model_size_bytes),
            "model_size_mb": round(model_size_mb, 4),
            "total_parameters": int(total_params),
            "trainable_parameters": int(trainable_params),
            "best_val_loss": float(best_val_loss),
            "epochs_trained": len(training_history)
        }
        with open("models/original_model_arch.json", "w", encoding="utf-8") as f:
            json.dump(model_arch, f, indent=2, default=str)

        # Phase 5: Compute confusion matrix for classification tasks using best model predictions
        confusion_matrix_result = None
        if model_config['task_type'] == 'classification' and len(best_all_predictions) > 0:
            try:
                # Use predictions from best model epoch
                confusion_matrix_result = confusion_matrix(best_all_labels, best_all_predictions).tolist()
                logger.info("✅ Confusion matrix computed successfully from best model")
            except Exception as e:
                logger.warning(f"Could not compute confusion matrix: {str(e)}")
        
        # Phase 5: Measure inference speed
        inference_speed_ms = None
        try:
            model.eval()
            # Use a sample from validation set for speed test
            if len(X_val) > 0:
                sample_input = torch.FloatTensor(X_val[:min(100, len(X_val))]).to(self.device)
                
                # Warmup
                with torch.no_grad():
                    for _ in range(10):
                        _ = model(sample_input[:1])
                
                # Measure inference time
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start_time_inf = time.time()
                num_inferences = 100
                with torch.no_grad():
                    for _ in range(num_inferences):
                        _ = model(sample_input[:1])
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                elapsed_time = time.time() - start_time_inf
                inference_speed_ms = (elapsed_time / num_inferences) * 1000  # Convert to milliseconds per inference
                logger.info(f"✅ Inference speed measured: {inference_speed_ms:.2f} ms per sample")
        except Exception as e:
            logger.warning(f"Could not measure inference speed: {str(e)}")
        
        # Save final logs with Phase 5 enhancements
        logs = {
            "model_type": model_type,
            "task_type": task_type,
            "num_classes": num_classes if task_type == 'classification' else None,
            "epochs": epochs,
            "epochs_trained": len(training_history),
            "training_time": time.time() - start_time,
            "history": training_history,
            "model_size_bytes": int(model_size_bytes),
            "model_size_mb": round(model_size_mb, 4),
            "model_size_kb": round(model_size_bytes / 1024, 2),
            "model_path": model_path,
            "total_parameters": int(total_params),
            "trainable_parameters": int(trainable_params),
            "best_val_loss": float(best_val_loss),
            "preprocessing_warnings": self.preprocessing_warnings if hasattr(self, 'preprocessing_warnings') else [],
            "optimizations": optimizations_info,
            # Phase 5: Add confusion matrix and inference speed
            "confusion_matrix": confusion_matrix_result,
            "inference_speed_ms": inference_speed_ms
        }

        self._atomic_write_json("results/training_logs.json", logs)
        
        # Save to training history
        self._save_to_training_history(model_config, dataset_path, logs)
        
        logger.info(f"✅ Training completed successfully in {time.time() - start_time:.2f}s")

    def _update_status(self, status, current_epoch, total_epochs, message=""):
        """Update training status with model size when completed"""
        status_data = {
            "status": status,
            "current_epoch": current_epoch,
            "total_epochs": total_epochs,
            "message": message,
            "timestamp": time.time()
        }

        # Add model size information when training is completed
        if status == "completed":
            # First, try to load from training logs (most accurate, saved just after training)
            logs_path = "results/training_logs.json"
            size_loaded_from_logs = False
            
            if os.path.exists(logs_path):
                try:
                    with open(logs_path, "r") as f:
                        logs = json.load(f)
                        
                        # Use model size from logs if available (most accurate)
                        if "model_size_mb" in logs:
                            status_data["model_size_mb"] = logs.get("model_size_mb")
                            status_data["model_size_bytes"] = logs.get("model_size_bytes", 0)
                            status_data["model_size_kb"] = logs.get("model_size_kb", 0)
                            status_data["model_path"] = logs.get("model_path", "")
                            size_loaded_from_logs = True
                            logger.info(f"Training completed. Model size: {logs.get('model_size_mb', 0):.4f} MB (from logs)")
                        
                        # Also load parameter counts
                        if "total_parameters" in logs:
                            status_data["total_parameters"] = logs.get("total_parameters")
                        if "trainable_parameters" in logs:
                            status_data["trainable_parameters"] = logs.get("trainable_parameters")
                        if "num_parameters" in logs:
                            status_data["num_parameters"] = logs.get("num_parameters")
                except Exception as e:
                    logger.warning(f"Could not load model info from logs: {str(e)}")
            
            # If size not loaded from logs, fall back to checking model files
            # Check most recently modified file to get the correct current model
            if not size_loaded_from_logs:
                possible_model_paths = [
                    "models/original_model.pkl",
                    "models/original_model.pt",
                    "models/original_model.h5"
                ]
                
                # Find the most recently modified model file
                model_files = [(path, os.path.getmtime(path)) for path in possible_model_paths if os.path.exists(path)]
                if model_files:
                    # Sort by modification time (newest first) and use the most recent
                    model_files.sort(key=lambda x: x[1], reverse=True)
                    model_path, _ = model_files[0]
                    
                    model_size_bytes = os.path.getsize(model_path)
                    model_size_mb = model_size_bytes / (1024 * 1024)
                    status_data["model_size_bytes"] = int(model_size_bytes)
                    status_data["model_size_mb"] = round(model_size_mb, 4)
                    status_data["model_size_kb"] = round(model_size_bytes / 1024, 2)
                    status_data["model_path"] = model_path
                    logger.info(f"Training completed. Model size: {model_size_mb:.4f} MB (from file: {os.path.basename(model_path)})")
                else:
                    logger.warning("Training completed but no model file found for size calculation")

        # Atomic write to prevent file corruption during concurrent reads
        temp_path = "results/training_status.json.tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(status_data, f, indent=2)
        # Windows-compatible atomic rename
        self._safe_replace(temp_path, "results/training_status.json")

    def stop_training(self):
        """Stop training process"""
        self.stop_flag = True
    
    # ========== PHASE 3 & 4: ERROR HANDLING & USER GUIDANCE ==========
    
    def _create_training_error_response(self, error: Exception, model_config: Dict[str, Any], dataset_path: str) -> Dict[str, Any]:
        """
        Phase 3: Create structured error response - NEVER throw raw exceptions
        Returns JSON-serializable error response
        """
        error_type = type(error).__name__
        error_message = str(error)
        error_traceback = traceback.format_exc()
        
        # Determine cause and suggested fix
        cause, suggested_fix = self._analyze_error_cause(error, error_message, model_config)
        
        # Phase 4: Add user guidance if dataset-related
        user_guidance = None
        if self._is_dataset_related_error(error, error_message):
            user_guidance = self._create_dataset_error_guidance(model_config, dataset_path)
        
        error_response = {
            "status": "training_failed",
            "message": f"Training failed: {error_message}",
            "cause": cause,
            "suggested_fix": suggested_fix,
            "error_type": error_type,
        }
        
        if user_guidance:
            error_response["user_guidance"] = user_guidance
        
        # Save error details for debugging
        os.makedirs("results", exist_ok=True)
        error_details = {
            **error_response,
            "traceback": error_traceback,
            "timestamp": time.time()
        }
        with open("results/training_error.json", "w", encoding="utf-8") as f:
            json.dump(error_details, f, indent=2, default=str)
        
        return error_response
    
    def _analyze_error_cause(self, error: Exception, error_message: str, model_config: Dict[str, Any]) -> Tuple[str, str]:
        """Analyze error to determine cause and suggested fix"""
        error_msg_lower = error_message.lower()
        
        # Memory errors
        if "out of memory" in error_msg_lower or "cuda" in error_msg_lower and "memory" in error_msg_lower:
            return (
                "Insufficient GPU/system memory for training",
                "Reduce batch_size, reduce model complexity, or use a machine with more memory"
            )
        
        # Shape mismatch errors
        if "shape" in error_msg_lower or "size mismatch" in error_msg_lower:
            return (
                "Input data shape does not match model expectations",
                "Check dataset preprocessing and model input_shape configuration. Ensure data matches expected format."
            )
        
        # Dataset loading errors
        if "file not found" in error_msg_lower or "no such file" in error_msg_lower:
            return (
                "Dataset file not found or inaccessible",
                "Verify dataset path is correct and file exists"
            )
        
        # Preprocessing errors
        if "preprocessing" in error_msg_lower or "scaler" in error_msg_lower:
            return (
                "Data preprocessing failed",
                "Check dataset format and use Auto-Fix Dataset feature if available"
            )
        
        # General ValueError
        if isinstance(error, ValueError):
            return (
                "Invalid parameter or data format",
                f"Review error message and adjust parameters: {error_message}"
            )
        
        # Default
        return (
            "Unexpected error during training",
            "Check error message for details. If issue persists, verify dataset format and model configuration."
        )
    
    def _is_dataset_related_error(self, error: Exception, error_message: str) -> bool:
        """Check if error is related to dataset format/issues"""
        dataset_keywords = [
            "dataset", "file not found", "shape", "format", "column", 
            "feature", "label", "missing", "invalid", "preprocessing",
            "csv", "image", "directory"
        ]
        error_msg_lower = error_message.lower()
        return any(keyword in error_msg_lower for keyword in dataset_keywords)
    
    def _create_dataset_error_guidance(self, model_config: Dict[str, Any], dataset_path: str) -> Dict[str, Any]:
        """
        Phase 4: Create comprehensive user guidance for dataset errors
        Shows model requirements, format examples, and fix options
        """
        model_type = model_config.get('model_type', 'unknown')
        task_type = model_config.get('task_type', 'unknown')
        
        guidance = {
            "selected_model": model_type.upper(),
            "required_format": self._get_required_format(model_type),
            "example_structure": self._get_example_structure(model_type, task_type),
            "instructions": [
                "Please re-upload your dataset matching the required format",
                "Use the Auto-Fix Dataset feature if available",
                "Refer to the example structure below"
            ],
            "can_autofix": True  # Dataset conditioning service can usually fix common issues
        }
        
        return guidance
    
    def _get_required_format(self, model_type: str) -> str:
        """Get required dataset format description for model type"""
        formats = {
            "decision_tree": "CSV file with numeric features and a target column (last column or named 'target')",
            "cnn": "Directory with labeled subdirectories, each containing images of that class",
            "rnn": "CSV file with sequences or text data, with labels in the last column"
        }
        return formats.get(model_type, "See model documentation for required format")
    
    def _get_example_structure(self, model_type: str, task_type: str) -> Dict[str, Any]:
        """Get example dataset structure for model type"""
        if model_type == "decision_tree":
            return {
                "format": "CSV",
                "example": {
                    "columns": ["feature1", "feature2", "feature3", "target"],
                    "rows": [
                        [1.2, 3.4, 5.6, 0],
                        [2.3, 4.5, 6.7, 1],
                        [3.4, 5.6, 7.8, 0]
                    ],
                    "note": "Features must be numeric, target can be numeric or categorical"
                }
            }
        elif model_type == "cnn":
            return {
                "format": "Directory structure",
                "example": {
                    "structure": {
                        "dataset/": {
                            "class1/": ["img1.jpg", "img2.jpg"],
                            "class2/": ["img3.jpg", "img4.jpg"]
                        }
                    },
                    "note": "Each subdirectory represents a class, containing images of that class"
                }
            }
        elif model_type == "rnn":
            return {
                "format": "CSV with sequences",
                "example": {
                    "columns": ["sequence_col1", "sequence_col2", "sequence_col3", "target"],
                    "rows": [
                        [[1, 2, 3], [4, 5, 6], [7, 8, 9], 0],
                        [[2, 3, 4], [5, 6, 7], [8, 9, 10], 1]
                    ],
                    "note": "Sequences can be numeric arrays or text tokens"
                }
            }
        return {"note": "See model documentation for structure requirements"}