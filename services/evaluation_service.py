"""
Evaluation Service - Handles model evaluation with comprehensive error handling
"""

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, mean_squared_error, r2_score
import pandas as pd
import numpy as np
import json
import time
import pickle
import os
import logging
from utils.model_builder import ModelBuilder
from utils.data_loader import DataLoaderUtil
from utils.validation import DataValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvaluationService:
    def __init__(self):
        self.model_builder = ModelBuilder()
        self.data_loader = DataLoaderUtil()
        self.validator = DataValidator()

    def evaluate(self, model_path, dataset_path, model_type):
        """Evaluate a model on test dataset with comprehensive validation"""
        try:
            # Validate model path exists
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Validate dataset path
            self.validator.validate_dataset_path(dataset_path)

            # Load model configuration - try multiple sources
            model_config = self._load_model_config()
            
            # Validate model config
            if not model_config:
                raise ValueError("Could not load model configuration. Please ensure model was trained first.")

            # Load test data - use the same dataset as training if possible
            # First check if we have training dataset path
            training_data_path = "results/training_data.json"
            use_training_dataset = False
            
            if os.path.exists(training_data_path):
                try:
                    with open(training_data_path, "r") as f:
                        training_data = json.load(f)
                    # Check if user is evaluating on same dataset as training
                    model_arch_path = "models/original_model_arch.json"
                    if os.path.exists(model_arch_path):
                        with open(model_arch_path, "r") as f:
                            arch_data = json.load(f)
                            training_dataset_path = arch_data.get("config", {}).get("dataset_path") or arch_data.get("dataset_path")
                            
                            # If user provided dataset path matches training dataset, use training data directly
                            if training_dataset_path and os.path.normpath(training_dataset_path) == os.path.normpath(dataset_path):
                                logger.info("Using training dataset for evaluation - using training split data")
                                X_test = np.array(training_data.get("X_val", []))
                                y_test = np.array(training_data.get("y_val", []))
                                use_training_dataset = True
                except Exception as e:
                    logger.warning(f"Could not load training data: {str(e)}")

            # Load from file if not using training data
            if not use_training_dataset:
                logger.info(f"Loading test data from {dataset_path}")
                X_test, y_test = self.data_loader.load_test_data(
                    dataset_path,
                    model_config.get('model_type', 'decision_tree')
                )
            
                # Validate test data
                if len(X_test) == 0:
                    raise ValueError("Test dataset is empty")
                if len(y_test) == 0:
                    raise ValueError("Test targets are empty")
            
                logger.info(f"Test data shape: X={X_test.shape}, y={y_test.shape}")

            # Evaluate based on model type
            if model_path.endswith('.pkl'):
                metrics = self._evaluate_sklearn_model(model_path, X_test, y_test, model_config)
            else:
                metrics = self._evaluate_pytorch_model(model_path, X_test, y_test, model_config)

            logger.info(f"Evaluation completed. Accuracy: {metrics.get('accuracy', 'N/A')}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}", exc_info=True)
            raise ValueError(f"Evaluation failed: {str(e)}") from e
    
    def _load_model_config(self):
        """Load model configuration from multiple possible sources"""
        config = None
        
        # Try architecture file first (most complete)
        arch_path = "models/original_model_arch.json"
        if os.path.exists(arch_path):
            try:
                with open(arch_path, "r") as f:
                    arch_data = json.load(f)
                    if "config" in arch_data:
                        config = arch_data["config"]
                    else:
                        config = arch_data
            except Exception as e:
                logger.warning(f"Could not load from architecture file: {str(e)}")
        
        # Fall back to selected model config
        if not config:
            config_path = "models/selected_model_config.json"
            if os.path.exists(config_path):
                try:
                    with open(config_path, "r") as f:
                        config = json.load(f)
                except Exception as e:
                    logger.warning(f"Could not load from selected config: {str(e)}")
        
        # Try to load from model metadata
        if not config or not config.get('task_type'):
            metadata_path = "models/original_model_metadata.json"
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                        if config:
                            config['task_type'] = metadata.get('task_type', config.get('task_type'))
                        else:
                            config = metadata
                except Exception as e:
                    logger.warning(f"Could not load model metadata: {str(e)}")
        
        # Ensure task_type is set if we have a config
        if config and not config.get('task_type'):
            # Try to infer from training data
            training_data_path = "results/training_data.json"
            if os.path.exists(training_data_path):
                try:
                    with open(training_data_path, "r") as f:
                        training_data = json.load(f)
                        y_train = training_data.get("y_train", [])
                        if y_train:
                            task_type, _ = self.validator.infer_task_type_from_target(np.array(y_train))
                            config['task_type'] = task_type
                            logger.info(f"Inferred task_type from training data: {task_type}")
                except Exception as e:
                    logger.warning(f"Could not infer task_type from training data: {str(e)}")
        
        return config

    def _evaluate_sklearn_model(self, model_path, X_test, y_test, model_config):
        """Evaluate sklearn model with enhanced error handling"""
        try:
            # Check for compressed model arch file to get task_type
            compressed_arch_paths = [
                model_path.replace('.pkl', '_arch.json'),
                model_path.replace('_model.pkl', '_model_arch.json'),
                "models/compressed_model_arch.json",
                "models/distilled_model_arch.json",
                "models/pruned_model_arch.json",
                "models/quantized_model_arch.json",
            ]
            
            for arch_path in compressed_arch_paths:
                if os.path.exists(arch_path):
                    try:
                        with open(arch_path, 'r') as f:
                            arch_data = json.load(f)
                        
                        # Update model_config with arch file data
                        if 'config' in arch_data:
                            if arch_data['config'].get('task_type'):
                                model_config['task_type'] = arch_data['config']['task_type']
                                logger.info(f"📁 Loaded task_type from arch file {arch_path}: {model_config['task_type']}")
                            break
                    except Exception as e:
                        logger.warning(f"Could not load arch file {arch_path}: {e}")
            
            # OPTIMIZATION: Sample subset for faster evaluation on large datasets
            max_eval_samples = 10000  # Increased for better accuracy
            if len(X_test) > max_eval_samples:
                import random
                random.seed(42)  # For reproducibility
                indices = random.sample(range(len(X_test)), max_eval_samples)
                X_test_eval = X_test[indices]
                y_test_eval = y_test[indices]
                logger.info(f"Evaluating on {max_eval_samples} samples (down from {len(X_test)})")
            else:
                X_test_eval = X_test
                y_test_eval = y_test
        
            # Load model metadata if available
            expected_n_features = None
            model_metadata_path = "models/original_model_metadata.json"
            if os.path.exists(model_metadata_path):
                try:
                    with open(model_metadata_path, "r") as f:
                        metadata = json.load(f)
                    expected_n_features = metadata.get("n_features")
                    if expected_n_features:
                        logger.info(f"Loaded model metadata: expecting {expected_n_features} features")
                except Exception as e:
                    logger.warning(f"Could not load model metadata: {str(e)}")
            
            # Load model (could be model+selector or just model)
            with open(model_path, "rb") as f:
                loaded_data = pickle.load(f)
        
            # Extract model and check feature count
            model = None
            selector = None
            
            # Handle different model save formats
            if isinstance(loaded_data, dict):
                # Check if it's compressed model with selector
                if "model" in loaded_data and "selector" in loaded_data:
                    model = loaded_data["model"]
                    selector = loaded_data["selector"]
                    # Get expected features from selector
                    if hasattr(selector, 'get_support'):
                        expected_n_features = selector.get_support().sum()
                    logger.info(f"Loaded compressed model with feature selector: expecting {expected_n_features} features")
                else:
                    # Unexpected dict format, try to extract model
                    model = loaded_data.get("model", loaded_data)
            else:
                # Regular model object
                model = loaded_data
                
                # Try to infer expected features from model if not in metadata
                if expected_n_features is None:
                    if hasattr(model, 'n_features_in_'):
                        expected_n_features = model.n_features_in_
                        logger.info(f"Inferred expected features from model: {expected_n_features}")
                    elif hasattr(model, 'feature_importances_'):
                        expected_n_features = len(model.feature_importances_)
                        logger.info(f"Inferred expected features from feature_importances: {expected_n_features}")
                
                # If still not found, try training data
                if expected_n_features is None:
                    training_data_path = "results/training_data.json"
                    if os.path.exists(training_data_path):
                        try:
                            with open(training_data_path, "r") as f:
                                training_data = json.load(f)
                            expected_n_features = training_data.get("n_features")
                            if expected_n_features:
                                logger.info(f"Got expected features from training data: {expected_n_features}")
                        except:
                            pass
            
            if model is None:
                raise ValueError("Loaded model is None")
            
            # Validate feature count before prediction
            actual_n_features = X_test_eval.shape[1]
            if expected_n_features is not None and actual_n_features != expected_n_features:
                logger.warning(f"Feature count mismatch: model expects {expected_n_features}, got {actual_n_features}")
                
                # Try to use the same dataset as training if available
                training_data_path = "results/training_data.json"
                if os.path.exists(training_data_path):
                    try:
                        with open(training_data_path, "r") as f:
                            training_data = json.load(f)
                        train_n_features = training_data.get("n_features")
                        
                        if train_n_features == expected_n_features:
                            logger.info("Feature count matches training data. Ensure you're using the same dataset as training.")
                        else:
                            raise ValueError(
                                f"Feature count mismatch: Model was trained with {expected_n_features} features, "
                                f"but test data has {actual_n_features} features. "
                                f"Please use the same dataset that was used for training, or re-train the model."
                            )
                    except Exception as e:
                        logger.error(f"Error checking training data: {str(e)}")
                
                if actual_n_features != expected_n_features:
                    raise ValueError(
                        f"Feature count mismatch: Model expects {expected_n_features} features, "
                        f"but test data has {actual_n_features} features. "
                        f"Please use the same dataset that was used for training."
                    )
            
            # Apply feature selector if present
            if selector is not None:
                try:
                    X_test_eval = selector.transform(X_test_eval)
                    logger.info(f"Applied feature selector: {X_test_eval.shape[1]} features remaining")
                except Exception as e:
                    logger.error(f"Feature selection transform failed: {str(e)}")
                    raise ValueError(f"Cannot apply feature selector: {str(e)}") from e

            # Measure inference time
            start_time = time.time()
            y_pred = model.predict(X_test_eval)
            inference_time = (time.time() - start_time) / len(X_test_eval)

            # Auto-detect task type if not reliable or missing
            # Check model type first
            detected_task_type = model_config.get('task_type')
            
            # Validate task type by checking model class
            if hasattr(model, '__class__'):
                model_class_name = model.__class__.__name__.lower()
                if 'regressor' in model_class_name:
                    detected_task_type = 'regression'
                    logger.info(f"Auto-detected regression from model class: {model_class_name}")
                elif 'classifier' in model_class_name:
                    detected_task_type = 'classification'
                    logger.info(f"Auto-detected classification from model class: {model_class_name}")
            
            # Validate by checking if predictions are continuous or discrete
            y_pred_array = np.array(y_pred)
            y_test_array = np.array(y_test_eval)
            
            # If predictions look continuous (float with decimals), it's likely regression
            if detected_task_type != 'regression':
                # Check if predictions are continuous
                if np.issubdtype(y_pred_array.dtype, np.floating):
                    unique_preds = len(np.unique(y_pred_array))
                    # If many unique values relative to samples, likely regression
                    if unique_preds > len(y_pred_array) * 0.9:
                        detected_task_type = 'regression'
                        logger.info(f"Auto-detected regression: {unique_preds} unique predictions from {len(y_pred_array)} samples")
            
            # If still not sure, check target values
            if not detected_task_type:
                if np.issubdtype(y_test_array.dtype, np.floating):
                    unique_targets = len(np.unique(y_test_array))
                    if unique_targets > len(y_test_array) * 0.8:
                        detected_task_type = 'regression'
                    else:
                        detected_task_type = 'classification'
                else:
                    detected_task_type = 'classification'
                logger.info(f"Auto-detected task type from target: {detected_task_type}")
            
            # Update model config with detected task type
            if detected_task_type and detected_task_type != model_config.get('task_type'):
                logger.warning(f"Task type mismatch: config says '{model_config.get('task_type')}', detected '{detected_task_type}'. Using detected type.")
                model_config['task_type'] = detected_task_type

            # Calculate metrics based on detected task type
            task_type = model_config.get('task_type', detected_task_type) or detected_task_type
            
            # Initialize all metrics variables
            accuracy = 0.0
            precision = 0.0
            recall = 0.0
            f1 = 0.0
            mse = 0.0
            r2 = 0.0
            conf_matrix = []
            
            if task_type == 'classification':
                try:
                    # Ensure predictions and targets are integers for classification
                    y_pred_int = np.round(y_pred_array).astype(int)
                    y_test_int = y_test_array.astype(int)
                    
                    accuracy = accuracy_score(y_test_int, y_pred_int)
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        y_test_int, y_pred_int, average='weighted', zero_division=0
                    )
                    conf_matrix = confusion_matrix(y_test_int, y_pred_int).tolist()
                except Exception as e:
                    logger.error(f"Classification metrics failed, trying regression: {str(e)}")
                    # Fallback to regression metrics
                    task_type = 'regression'
                    model_config['task_type'] = 'regression'
                    mse = mean_squared_error(y_test_array, y_pred_array)
                    r2 = r2_score(y_test_array, y_pred_array)
                    accuracy = r2  # Use R2 as "accuracy" for regression
                    precision, recall, f1 = 0, 0, 0
                    conf_matrix = []
            else:  # regression
                mse = mean_squared_error(y_test_array, y_pred_array)
                r2 = r2_score(y_test_array, y_pred_array)
                accuracy = r2  # Use R2 score as accuracy for regression
                precision, recall, f1 = 0, 0, 0
                conf_matrix = []

            metrics = {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "inference_time": float(inference_time),
                "confusion_matrix": conf_matrix if conf_matrix else None,
                "model_type": model_config.get('model_type', 'unknown'),
                "task_type": task_type
            }

            if task_type == 'regression':
                metrics['mse'] = float(mse)
                metrics['r2_score'] = float(r2)
                metrics['accuracy'] = float(r2)  # For regression, accuracy is R2

            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating sklearn model: {str(e)}", exc_info=True)
            raise

    def _evaluate_pytorch_model(self, model_path, X_test, y_test, model_config):
        """Evaluate PyTorch model with enhanced error handling"""
        try:
            # Determine which config to use - check for compressed model arch file first
            effective_config = model_config.copy()
            
            # Check if this is a compressed/distilled model - look for its specific arch file
            compressed_arch_paths = [
                model_path.replace('.pt', '_arch.json'),  # distilled_model_arch.json
                model_path.replace('_model.pt', '_model_arch.json'),  # compressed_model_arch.json
                "models/distilled_model_arch.json",  # Distillation output
                "models/compressed_model_arch.json",  # Generic compressed
            ]
            
            for arch_path in compressed_arch_paths:
                if os.path.exists(arch_path):
                    try:
                        with open(arch_path, 'r') as f:
                            arch_data = json.load(f)
                        
                        # Use the config from arch file (has correct architecture for compressed model)
                        if 'config' in arch_data:
                            effective_config = arch_data['config']
                            logger.info(f"📁 Loaded compressed model config from {arch_path}")
                            break
                    except Exception as e:
                        logger.warning(f"Could not load arch file {arch_path}: {e}")
            
            # Validate config has required fields
            if not effective_config.get('num_classes'):
                effective_config['num_classes'] = model_config.get('num_classes', 1)
            if effective_config.get('model_type') in ['cnn', 'rnn'] and not effective_config.get('input_shape'):
                effective_config['input_shape'] = model_config.get('input_shape')
            
            # Build model with the correct (possibly compressed) config
            logger.info(f"📦 Building model with config: model_type={effective_config.get('model_type')}, hidden={effective_config.get('config', {}).get('hidden_size', 'default')}")
            model = self.model_builder.build_pytorch_model(effective_config)

            # Load model weights
            if model_path.endswith('.pt'):
                try:
                    state_dict = torch.load(model_path, map_location='cpu')
                    model.load_state_dict(state_dict)
                    logger.info(f"✅ Loaded model weights from {model_path}")
                except Exception as e:
                    logger.error(f"Error loading model state dict: {str(e)}")
                    raise ValueError(f"Cannot load model weights: {str(e)}") from e
            else:
                raise FileNotFoundError(f"Model file not found: {model_path}")

            model.eval()

            # OPTIMIZATION: Sample subset for faster evaluation on large datasets
            max_eval_samples = 10000  # Increased for better accuracy
            if len(X_test) > max_eval_samples:
                import random
                random.seed(42)  # For reproducibility
                indices = random.sample(range(len(X_test)), max_eval_samples)
                X_test_eval = X_test[indices]
                y_test_eval = y_test[indices]
                logger.info(f"Evaluating on {max_eval_samples} samples (down from {len(X_test)})")
            else:
                X_test_eval = X_test
                y_test_eval = y_test

            # Prepare data - detect task type first to use correct tensor type
            task_type_for_tensor = model_config.get('task_type', 'classification')
            
            # Try to infer task type from y_test if not set
            if not task_type_for_tensor:
                y_test_array = np.array(y_test_eval)
                if np.issubdtype(y_test_array.dtype, np.floating) and len(np.unique(y_test_array)) > len(y_test_array) * 0.8:
                    task_type_for_tensor = 'regression'
                else:
                    task_type_for_tensor = 'classification'
            
            X_test_tensor = torch.FloatTensor(X_test_eval)
            y_test_tensor = torch.LongTensor(y_test_eval) if task_type_for_tensor == 'classification' else torch.FloatTensor(y_test_eval)

            # Measure inference time
            start_time = time.time()
            with torch.no_grad():
                outputs = model(X_test_tensor)
            inference_time = (time.time() - start_time) / len(X_test_eval)

            # Auto-detect task type if needed
            task_type = model_config.get('task_type')
            
            # Validate task type from model outputs
            outputs_array = outputs.detach().cpu().numpy()
            y_true_array = y_test_tensor.detach().cpu().numpy()
            
            # Check if it's classification (outputs have multiple dimensions) or regression (single output)
            if outputs_array.ndim == 2 and outputs_array.shape[1] > 1:
                # Multiple outputs - likely classification
                if task_type != 'classification':
                    logger.info(f"Model outputs suggest classification (shape: {outputs_array.shape}), updating task_type")
                    task_type = 'classification'
            elif outputs_array.ndim == 1 or (outputs_array.ndim == 2 and outputs_array.shape[1] == 1):
                # Single output - likely regression
                if task_type != 'regression':
                    logger.info(f"Model outputs suggest regression (shape: {outputs_array.shape}), updating task_type")
                    task_type = 'regression'
            
            # Ensure task_type is set
            if not task_type:
                task_type = 'classification'  # Default fallback
                logger.warning("Task type not detected, defaulting to classification")
            
            model_config['task_type'] = task_type

            # Initialize all metrics variables
            accuracy = 0.0
            precision = 0.0
            recall = 0.0
            f1 = 0.0
            mse = 0.0
            r2 = 0.0
            loss = 0.0
            conf_matrix = []
            
            # Calculate metrics based on task type
            if task_type == 'classification':
                try:
                    _, predicted = torch.max(outputs.data, 1)
                    y_pred = predicted.detach().cpu().numpy()
                    y_true = y_true_array

                    accuracy = accuracy_score(y_true, y_pred)
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        y_true, y_pred, average='weighted', zero_division=0
                    )
                    conf_matrix = confusion_matrix(y_true, y_pred).tolist()

                    # Calculate loss
                    criterion = nn.CrossEntropyLoss()
                    loss = criterion(outputs, y_test_tensor).item()
                except Exception as e:
                    logger.error(f"Classification metrics failed: {str(e)}, trying regression metrics")
                    # Fallback to regression
                    task_type = 'regression'
                    y_pred = outputs_array.flatten() if outputs_array.ndim > 1 else outputs_array
                    y_true = y_true_array.flatten() if y_true_array.ndim > 1 else y_true_array

                    mse = mean_squared_error(y_true, y_pred)
                    r2 = r2_score(y_true, y_pred)

                    accuracy = r2
                    precision, recall, f1 = 0, 0, 0
                    conf_matrix = []
                    loss = mse
            else:  # regression
                y_pred = outputs_array.flatten() if outputs_array.ndim > 1 else outputs_array
                y_true = y_true_array.flatten() if y_true_array.ndim > 1 else y_true_array

                mse = mean_squared_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)

                accuracy = r2
                precision, recall, f1 = 0, 0, 0
                conf_matrix = []
                loss = mse

            metrics = {
                "accuracy": float(accuracy),
                "loss": float(loss),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "inference_time": float(inference_time),
                "confusion_matrix": conf_matrix if conf_matrix else None,
                "model_type": model_config.get('model_type', 'unknown'),
                "task_type": task_type
            }

            if task_type == 'regression':
                metrics['mse'] = float(mse)
                metrics['r2_score'] = float(r2)
                metrics['accuracy'] = float(r2)  # For regression, accuracy is R2

            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating PyTorch model: {str(e)}", exc_info=True)
            raise