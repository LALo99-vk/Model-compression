"""
Data Loader - Handles data loading and preprocessing
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from PIL import Image
import os
import json
import pickle
import glob


class DataLoaderUtil:
    """Utility for loading and preprocessing data"""

    def load_data(self, dataset_path, model_type, validation_split=0.2):
        """Load and split data for training"""

        if dataset_path.endswith('.csv'):
            return self._load_csv_data(dataset_path, model_type, validation_split)
        else:
            return self._load_image_data(dataset_path, model_type, validation_split)

    def load_test_data(self, dataset_path, model_type):
        """Load test data with chunking for large datasets"""

        if dataset_path.endswith('.csv'):
            # Check file size to determine if we need chunking
            file_size_mb = os.path.getsize(dataset_path) / (1024 * 1024)
            
            if file_size_mb > 50:  # Use chunking for files > 50MB
                return self._load_large_test_data(dataset_path, model_type)
            else:
                return self._load_small_test_data(dataset_path, model_type)
        else:
            # Load images
            return self._load_image_data_test(dataset_path)
    
    def _load_small_test_data(self, dataset_path, model_type):
        """Load small test data normally"""
        # Load CSV and assume last column is target
        df = pd.read_csv(dataset_path)

        # Separate features and target
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        # Encode labels if classification
        if model_type != 'decision_tree' or np.issubdtype(y.dtype, np.number):
            if not np.issubdtype(y.dtype, np.number):
                le = LabelEncoder()
                y = le.fit_transform(y)

        # Load saved scaler if it exists
        if os.path.exists("models/scaler.pkl"):
            with open("models/scaler.pkl", "rb") as f:
                scaler = pickle.load(f)
            X = scaler.transform(X)  # Use transform, not fit_transform
        else:
            # Fallback: fit new scaler
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        # Reshape for RNN if needed
        if model_type == 'rnn':
            # Reshape to (samples, timesteps, features)
            X = X.reshape(X.shape[0], -1, 1)

        return X, y
    
    def _load_large_test_data(self, dataset_path, model_type):
        """Load large test data with chunking and sampling"""
        import os
        
        print(f"Loading large test dataset ({os.path.getsize(dataset_path) / (1024 * 1024):.1f}MB) with chunking...")
        
        # Check if we have dataset info from training
        dataset_info_path = "models/dataset_info.json"
        if os.path.exists(dataset_info_path):
            with open(dataset_info_path, "r") as f:
                dataset_info = json.load(f)
            print(f"Found training info: {dataset_info}")
        
        # Sample test data to keep memory manageable - REDUCED for faster evaluation
        chunk_size = 10000
        test_samples = 10000  # Reduced from 50k to 10k for faster evaluation
        sample_data = []
        
        for chunk in pd.read_csv(dataset_path, chunksize=chunk_size):
            # Sample from each chunk
            remaining_samples = test_samples - len(sample_data)
            if remaining_samples <= 0:
                break
                
            sample_size = min(len(chunk), remaining_samples)
            sample_chunk = chunk.sample(sample_size, random_state=42)
            sample_data.append(sample_chunk)
        
        # Combine samples
        sample_df = pd.concat(sample_data, ignore_index=True)
        print(f"Sampled {len(sample_df)} rows for testing")
        
        # Process sampled data
        X = sample_df.iloc[:, :-1].values
        y = sample_df.iloc[:, -1].values

        # Encode labels if necessary
        if not np.issubdtype(y.dtype, np.number):
            le = LabelEncoder()
            y = le.fit_transform(y)

        # Load saved scaler if it exists
        if os.path.exists("models/scaler.pkl"):
            with open("models/scaler.pkl", "rb") as f:
                scaler = pickle.load(f)
            X = scaler.transform(X)  # Use transform, not fit_transform
        else:
            # Fallback: fit new scaler
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        # Reshape for RNN if needed
        if model_type == 'rnn':
            X = X.reshape(X.shape[0], -1, 1)

        return X, y

    def _load_csv_data(self, csv_path, model_type, validation_split):
        """Load CSV data with chunking for large datasets"""
        # Check file size to determine if we need chunking
        file_size_mb = os.path.getsize(csv_path) / (1024 * 1024)
        
        if file_size_mb > 50:  # Use chunking for files > 50MB
            return self._load_large_csv_data(csv_path, model_type, validation_split)
        else:
            return self._load_small_csv_data(csv_path, model_type, validation_split)
    
    def _load_small_csv_data(self, csv_path, model_type, validation_split):
        """Load small CSV data normally"""
        df = pd.read_csv(csv_path)

        # Assume last column is target
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        # Encode labels if necessary
        if not np.issubdtype(y.dtype, np.number):
            le = LabelEncoder()
            y = le.fit_transform(y)

        # Scale features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )

        # Save scaler for later use
        with open("models/scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

        return X_train, X_val, y_train, y_val
    
    def _load_large_csv_data(self, csv_path, model_type, validation_split):
        """Load large CSV data with chunking and sampling"""
        import os
        
        print(f"Loading large dataset ({os.path.getsize(csv_path) / (1024 * 1024):.1f}MB) with chunking...")
        
        # First pass: determine dataset size and sample for training
        chunk_size = 10000
        total_rows = 0
        sample_data = []
        
        for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
            total_rows += len(chunk)
            # Sample 5% of each chunk for training (REDUCED for faster training)
            if len(sample_data) < 50000:  # Reduced from 100k to 50k samples for training
                sample_chunk = chunk.sample(min(len(chunk), chunk_size // 20), random_state=42)  # Reduced from 10% to 5%
                sample_data.append(sample_chunk)
        
        # Combine samples
        sample_df = pd.concat(sample_data, ignore_index=True)
        print(f"Sampled {len(sample_df)} rows from {total_rows} total rows for training")
        
        # Process sampled data
        X = sample_df.iloc[:, :-1].values
        y = sample_df.iloc[:, -1].values

        # Encode labels if necessary
        if not np.issubdtype(y.dtype, np.number):
            le = LabelEncoder()
            y = le.fit_transform(y)

        # Scale features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )

        # Save scaler and dataset info for later use
        with open("models/scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        
        dataset_info = {
            "total_rows": total_rows,
            "sampled_rows": len(sample_df),
            "chunk_size": chunk_size,
            "original_csv_path": csv_path
        }
        with open("models/dataset_info.json", "w") as f:
            json.dump(dataset_info, f, indent=2)

        return X_train, X_val, y_train, y_val

    def _load_image_data(self, image_dir, model_type, validation_split):
        """Load image data from directory"""

        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(image_dir, ext)))
            image_files.extend(glob.glob(os.path.join(image_dir, '**', ext), recursive=True))

        if not image_files:
            raise ValueError(f"No images found in {image_dir}")

        # Load images and extract labels from filenames or directory structure
        X = []
        y = []

        for img_path in image_files:
            # Load and preprocess image
            img = Image.open(img_path).convert('RGB')
            img = img.resize((32, 32))  # Resize to standard size
            img_array = np.array(img) / 255.0  # Normalize
            img_array = img_array.transpose(2, 0, 1)  # (C, H, W) for PyTorch
            X.append(img_array)

            # Extract label from directory structure or filename
            # Assume images are in subdirectories named by class
            label = os.path.basename(os.path.dirname(img_path))
            y.append(label)

        X = np.array(X, dtype=np.float32)

        # Encode labels
        le = LabelEncoder()
        y = le.fit_transform(y)

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )

        return X_train, X_val, y_train, y_val

    def _load_image_data_test(self, image_dir):
        """Load test images"""

        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(image_dir, ext)))
            image_files.extend(glob.glob(os.path.join(image_dir, '**', ext), recursive=True))

        X = []
        y = []

        for img_path in image_files:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((32, 32))
            img_array = np.array(img) / 255.0
            img_array = img_array.transpose(2, 0, 1)
            X.append(img_array)

            label = os.path.basename(os.path.dirname(img_path))
            y.append(label)

        X = np.array(X, dtype=np.float32)

        le = LabelEncoder()
        y = le.fit_transform(y)

        return X, y