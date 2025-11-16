"""
Training Service - Handles model training logic
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import json
import time
import pickle
from utils.model_builder import ModelBuilder
from utils.data_loader import DataLoaderUtil


class TrainingService:
    def __init__(self):
        self.stop_flag = False
        self.model_builder = ModelBuilder()
        self.data_loader = DataLoaderUtil()

    def train_model(self, model_config, dataset_path, epochs, batch_size, validation_split):
        """Train model based on configuration"""
        self.stop_flag = False

        try:
            # Update status
            self._update_status("loading_data", 0, epochs)

            # Load data
            X_train, X_val, y_train, y_val = self.data_loader.load_data(
                dataset_path,
                model_config['model_type'],
                validation_split
            )

            model_type = model_config['model_type']

            # Train based on model type
            if model_type == 'decision_tree':
                self._train_sklearn_model(model_config, X_train, X_val, y_train, y_val)
            elif model_type in ['cnn', 'rnn']:
                self._train_pytorch_model(model_config, X_train, X_val, y_train, y_val, epochs, batch_size)

            # Update final status
            self._update_status("completed", epochs, epochs)

        except Exception as e:
            self._update_status("error", 0, epochs, str(e))
            raise

    def _train_sklearn_model(self, model_config, X_train, X_val, y_train, y_val):
        """Train sklearn-based models"""
        config = model_config['config']
        task_type = model_config['task_type']

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
        self._update_status("training", 0, 1)
        model.fit(X_train, y_train)

        # Validate
        train_score = model.score(X_train, y_train)
        val_score = model.score(X_val, y_val)

        # Save model
        with open("models/original_model.pkl", "wb") as f:
            pickle.dump(model, f)

        # Save training data for compression
        training_data = {
            "X_train": X_train.tolist() if hasattr(X_train, 'tolist') else X_train,
            "y_train": y_train.tolist() if hasattr(y_train, 'tolist') else y_train,
            "X_val": X_val.tolist() if hasattr(X_val, 'tolist') else X_val,
            "y_val": y_val.tolist() if hasattr(y_val, 'tolist') else y_val
        }
        with open("results/training_data.json", "w") as f:
            json.dump(training_data, f, indent=2)

        # Save logs
        logs = {
            "model_type": "decision_tree",
            "train_score": float(train_score),
            "val_score": float(val_score),
            "epochs": 1,
            "training_time": time.time()
        }

        with open("results/training_logs.json", "w") as f:
            json.dump(logs, f, indent=2)

    def _train_pytorch_model(self, model_config, X_train, X_val, y_train, y_val, epochs, batch_size):
        """Train PyTorch models"""
        # Build model
        model = self.model_builder.build_pytorch_model(model_config)

        # Prepare data
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train) if model_config['task_type'] == 'classification' else torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val) if model_config['task_type'] == 'classification' else torch.FloatTensor(y_val)
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Setup training
        if model_config['task_type'] == 'classification':
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()

        optimizer = optim.Adam(model.parameters(), lr=model_config['config'].get('learning_rate', 0.001))

        # Training loop
        training_history = []

        for epoch in range(epochs):
            if self.stop_flag:
                break

            model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()

                    if model_config['task_type'] == 'classification':
                        _, predicted = torch.max(outputs.data, 1)
                        total += batch_y.size(0)
                        correct += (predicted == batch_y).sum().item()

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = correct / total if total > 0 else 0

            # Log epoch
            epoch_log = {
                "epoch": epoch + 1,
                "train_loss": float(avg_train_loss),
                "val_loss": float(avg_val_loss),
                "val_accuracy": float(val_accuracy)
            }
            training_history.append(epoch_log)

            # Update status
            self._update_status("training", epoch + 1, epochs,
                                f"Loss: {avg_train_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        # Save model
        torch.save(model.state_dict(), "models/original_model.pt")

        # Save model architecture for later loading
        model_arch = {
            "config": model_config,
            "state_dict_path": "models/original_model.pt"
        }
        with open("models/original_model_arch.json", "w") as f:
            json.dump(model_arch, f, indent=2)

        # Save logs
        logs = {
            "model_type": model_config['model_type'],
            "epochs": epochs,
            "history": training_history
        }

        with open("results/training_logs.json", "w") as f:
            json.dump(logs, f, indent=2)

    def _update_status(self, status, current_epoch, total_epochs, message=""):
        """Update training status"""
        status_data = {
            "status": status,
            "current_epoch": current_epoch,
            "total_epochs": total_epochs,
            "message": message,
            "timestamp": time.time()
        }

        with open("results/training_status.json", "w") as f:
            json.dump(status_data, f, indent=2)

    def stop_training(self):
        """Stop training process"""
        self.stop_flag = True