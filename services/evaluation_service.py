"""
Evaluation Service - Handles model evaluation
"""

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pandas as pd
import numpy as np
import json
import time
import pickle
import os
from utils.model_builder import ModelBuilder
from utils.data_loader import DataLoaderUtil


class EvaluationService:
    def __init__(self):
        self.model_builder = ModelBuilder()
        self.data_loader = DataLoaderUtil()

    def evaluate(self, model_path, dataset_path, model_type):
        """Evaluate a model on test dataset"""

        # Load model configuration
        config_path = "models/selected_model_config.json"
        with open(config_path, "r") as f:
            model_config = json.load(f)

        # Load test data
        X_test, y_test = self.data_loader.load_test_data(
            dataset_path,
            model_config['model_type']
        )

        # Evaluate based on model type
        if model_path.endswith('.pkl'):
            metrics = self._evaluate_sklearn_model(model_path, X_test, y_test, model_config)
        else:
            metrics = self._evaluate_pytorch_model(model_path, X_test, y_test, model_config)

        return metrics

    def _evaluate_sklearn_model(self, model_path, X_test, y_test, model_config):
        """Evaluate sklearn model"""
        # OPTIMIZATION: Sample subset for faster evaluation on large datasets
        max_eval_samples = 5000  # Cap at 5k samples for faster evaluation
        if len(X_test) > max_eval_samples:
            import random
            indices = random.sample(range(len(X_test)), max_eval_samples)
            X_test_eval = X_test[indices]
            y_test_eval = y_test[indices]
            print(f"Evaluating on {max_eval_samples} samples (down from {len(X_test)})")
        else:
            X_test_eval = X_test
            y_test_eval = y_test
        
        # Load model (could be just model or model+selector)
        with open(model_path, "rb") as f:
            loaded_data = pickle.load(f)
        
        # Handle compressed models with feature selectors
        if isinstance(loaded_data, dict) and "model" in loaded_data and "selector" in loaded_data:
            # Compressed model with feature selection
            model = loaded_data["model"]
            selector = loaded_data["selector"]
            
            # Apply feature selection to test data
            X_test_eval = selector.transform(X_test_eval)
        else:
            # Regular model
            model = loaded_data

        # Measure inference time
        start_time = time.time()
        y_pred = model.predict(X_test_eval)
        inference_time = (time.time() - start_time) / len(X_test_eval)

        # Calculate metrics
        accuracy = accuracy_score(y_test_eval, y_pred)

        if model_config['task_type'] == 'classification':
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test_eval, y_pred, average='weighted', zero_division=0
            )
            conf_matrix = confusion_matrix(y_test_eval, y_pred).tolist()
        else:
            from sklearn.metrics import mean_squared_error, r2_score
            mse = mean_squared_error(y_test_eval, y_pred)
            r2 = r2_score(y_test_eval, y_pred)
            precision, recall, f1 = 0, 0, 0
            conf_matrix = []

        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "inference_time": float(inference_time),
            "confusion_matrix": conf_matrix if conf_matrix else None,
            "model_type": model_config['model_type'],
            "task_type": model_config['task_type']
        }

        if model_config['task_type'] == 'regression':
            metrics['mse'] = float(mse)
            metrics['r2_score'] = float(r2)

        return metrics

    def _evaluate_pytorch_model(self, model_path, X_test, y_test, model_config):
        """Evaluate PyTorch model"""
        # Load model
        model = self.model_builder.build_pytorch_model(model_config)

        # Handle different model path extensions
        if model_path.endswith('.pt'):
            model.load_state_dict(torch.load(model_path))
        elif os.path.exists(model_path.replace('_model.pt', '_model_arch.json')):
            # Load from architecture file if state dict doesn't exist
            arch_path = model_path.replace('_model.pt', '_model_arch.json')
            with open(arch_path, 'r') as f:
                arch = json.load(f)
            state_dict_path = arch.get('state_dict_path', model_path)
            if os.path.exists(state_dict_path):
                model.load_state_dict(torch.load(state_dict_path))

        model.eval()

        # OPTIMIZATION: Sample subset for faster evaluation on large datasets
        max_eval_samples = 5000  # Cap at 5k samples for faster evaluation
        if len(X_test) > max_eval_samples:
            import random
            indices = random.sample(range(len(X_test)), max_eval_samples)
            X_test_eval = X_test[indices]
            y_test_eval = y_test[indices]
            print(f"Evaluating on {max_eval_samples} samples (down from {len(X_test)})")
        else:
            X_test_eval = X_test
            y_test_eval = y_test

        # Prepare data
        X_test_tensor = torch.FloatTensor(X_test_eval)
        y_test_tensor = torch.LongTensor(y_test_eval) if model_config[
                                                        'task_type'] == 'classification' else torch.FloatTensor(y_test_eval)

        # Measure inference time
        start_time = time.time()
        with torch.no_grad():
            outputs = model(X_test_tensor)
        inference_time = (time.time() - start_time) / len(X_test_eval)

        # Calculate metrics
        if model_config['task_type'] == 'classification':
            _, predicted = torch.max(outputs.data, 1)
            y_pred = predicted.numpy()
            y_true = y_test_tensor.numpy()

            accuracy = accuracy_score(y_true, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='weighted', zero_division=0
            )
            conf_matrix = confusion_matrix(y_true, y_pred).tolist()

            # Calculate loss
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, y_test_tensor).item()
        else:
            y_pred = outputs.numpy()
            y_true = y_test_tensor.numpy()

            from sklearn.metrics import mean_squared_error, r2_score
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
            "model_type": model_config['model_type'],
            "task_type": model_config['task_type']
        }

        if model_config['task_type'] == 'regression':
            metrics['mse'] = float(mse)
            metrics['r2_score'] = float(r2)

        return metrics