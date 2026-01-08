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
import logging
from typing import Tuple, Dict, Any, Optional
from utils.validation import DataValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoaderUtil:
    """Utility for loading and preprocessing data"""

    def __init__(self):
        self.validator = DataValidator()
        self.label_encoder = None  # Store label encoder for consistent encoding

    def load_data(self, dataset_path, model_type, validation_split=0.2):
        """Load and split data for training with comprehensive validation"""
        try:
            # Validate dataset path
            self.validator.validate_dataset_path(dataset_path)
            
            # Detect dataset type
            dataset_type = self.validator.detect_dataset_type(dataset_path)

            if dataset_type == 'csv':
                return self._load_csv_data(dataset_path, model_type, validation_split)
            else:
                return self._load_image_data(dataset_path, model_type, validation_split)
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_path}: {str(e)}")
            raise ValueError(f"Failed to load dataset: {str(e)}") from e

    def load_test_data(self, dataset_path, model_type):
        """Load test data with chunking for large datasets"""
        try:
            # Validate dataset path
            self.validator.validate_dataset_path(dataset_path)
            
            # Detect dataset type
            dataset_type = self.validator.detect_dataset_type(dataset_path)

            if dataset_type == 'csv':
                # Check file size to determine if we need chunking
                file_size_mb = os.path.getsize(dataset_path) / (1024 * 1024)
                
                if file_size_mb > 50:  # Use chunking for files > 50MB
                    return self._load_large_test_data(dataset_path, model_type)
                else:
                    return self._load_small_test_data(dataset_path, model_type)
            else:
                # Load images
                return self._load_image_data_test(dataset_path)
        except Exception as e:
            logger.error(f"Error loading test data from {dataset_path}: {str(e)}")
            raise ValueError(f"Failed to load test data: {str(e)}") from e
    
    def _load_small_test_data(self, dataset_path, model_type):
        """Load small test data with validation"""
        try:
            # Load CSV
            try:
                df = pd.read_csv(dataset_path, encoding='utf-8')
            except UnicodeDecodeError:
                logger.warning(f"UTF-8 encoding failed, trying latin-1")
                df = pd.read_csv(dataset_path, encoding='latin-1')
            
            # Validate structure
            metadata = self.validator.validate_csv_structure(df, dataset_path)
            
            # Prepare features
            df_clean = self.validator.prepare_features(df, handle_missing='drop')

        # Separate features and target
            target_col = metadata['target_column']
            X = df_clean.drop(columns=[target_col]).values
            y = df_clean[target_col].values

            # Load saved label encoder if exists
            if os.path.exists("models/label_encoder.pkl"):
                with open("models/label_encoder.pkl", "rb") as f:
                    self.label_encoder = pickle.load(f)
            if not np.issubdtype(y.dtype, np.number):
                    try:
                        y = self.label_encoder.transform(y)
                    except ValueError:
                        logger.warning("Label encoder mismatch, fitting new encoder")
                        self.label_encoder = LabelEncoder()
                        y = self.label_encoder.fit_transform(y)
            elif not np.issubdtype(y.dtype, np.number):
                self.label_encoder = LabelEncoder()
                y = self.label_encoder.fit_transform(y)

            # Load saved scaler if it exists
            if os.path.exists("models/scaler.pkl"):
                with open("models/scaler.pkl", "rb") as f:
                    scaler = pickle.load(f)
                X = scaler.transform(X)  # Use transform, not fit_transform
            else:
                logger.warning("Scaler not found, fitting new scaler on test data")
                scaler = StandardScaler()
                X = scaler.fit_transform(X)

            # Reshape for RNN if needed
            if model_type == 'rnn':
                # Reshape to (samples, timesteps, features)
                if X.ndim == 2:
                    X = X.reshape(X.shape[0], -1, 1)

            logger.info(f"Loaded {len(X)} test samples")
            return X, y
            
        except Exception as e:
            logger.error(f"Error loading small test data: {str(e)}")
            raise
    
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
        """Load small CSV data with validation and preprocessing"""
        try:
            # Load CSV with error handling
            try:
                df = pd.read_csv(csv_path, encoding='utf-8')
            except UnicodeDecodeError:
                logger.warning(f"UTF-8 encoding failed for {csv_path}, trying latin-1")
                df = pd.read_csv(csv_path, encoding='latin-1')

            # Validate dataset structure
            metadata = self.validator.validate_csv_structure(df, csv_path)
            logger.info(f"Dataset metadata: {metadata}")
            
            # Prepare features (handle missing values)
            df_clean = self.validator.prepare_features(df, handle_missing='drop')
            
            # Separate features and target
            target_col = metadata['target_column']
            feature_df = df_clean.drop(columns=[target_col])
            
            # Convert categorical features to numeric before checking data quality
            categorical_features = metadata.get('categorical_features', [])
            if categorical_features:
                from sklearn.preprocessing import LabelEncoder
                for col in categorical_features:
                    if col in feature_df.columns:
                        le = LabelEncoder()
                        feature_df[col] = le.fit_transform(feature_df[col].astype(str))
                        logger.info(f"Encoded categorical feature '{col}' to numeric")
            
            # Ensure all remaining columns are numeric (convert if possible, drop if not)
            numeric_df = feature_df.select_dtypes(include=[np.number])
            non_numeric_cols = feature_df.select_dtypes(exclude=[np.number]).columns.tolist()
            if non_numeric_cols:
                logger.warning(f"Dropping non-numeric columns that couldn't be encoded: {non_numeric_cols}")
                feature_df = numeric_df
            
            X = feature_df.values
            y = df_clean[target_col].values
            
            # Ensure X is numeric before checking data quality
            if not np.issubdtype(X.dtype, np.number):
                # Try to convert to numeric
                try:
                    X = pd.DataFrame(X).apply(pd.to_numeric, errors='coerce').values
                    # Remove any columns that couldn't be converted
                    X = X[:, ~np.isnan(X).all(axis=0)]
                    if X.shape[1] == 0:
                        raise ValueError("No numeric features found after conversion")
                except Exception as e:
                    raise ValueError(f"Features contain non-numeric data that cannot be converted: {str(e)}")
            
            # Check data quality
            quality_check = self.validator.check_data_quality(X, y)
            if quality_check['issues']:
                logger.warning(f"Data quality issues: {quality_check['issues']}")
            if quality_check['warnings']:
                logger.info(f"Data quality warnings: {quality_check['warnings']}")

            # Encode labels if necessary
            if not np.issubdtype(y.dtype, np.number):
                self.label_encoder = LabelEncoder()
                y = self.label_encoder.fit_transform(y)
                # Save label encoder
                with open("models/label_encoder.pkl", "wb") as f:
                    pickle.dump(self.label_encoder, f)
            
            # All features should now be numeric after conversion above
            # Scale features (only numeric features should remain)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Validate validation split
            if validation_split <= 0 or validation_split >= 1:
                logger.warning(f"Invalid validation_split {validation_split}, using 0.2")
                validation_split = 0.2
            
            # Ensure we have enough samples after split
            min_samples = max(10, int(1 / validation_split) + 1)
            if len(X_scaled) < min_samples:
                raise ValueError(f"Insufficient samples for validation split. "
                               f"Need at least {min_samples}, got {len(X_scaled)}")

            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y, test_size=validation_split, random_state=42, stratify=y if metadata['task_type'] == 'classification' else None
            )
            
            # Update metadata with actual feature count after preprocessing
            metadata['num_features'] = X.shape[1]
            metadata['feature_names'] = feature_df.columns.tolist()[:X.shape[1]]
            
            # Save metadata for later use
            dataset_metadata = {
                'path': csv_path,
                'metadata': metadata,
                'quality_check': quality_check,
                'feature_count': X.shape[1],
                'sample_count': len(X),
                'train_samples': len(X_train),
                'val_samples': len(X_val)
            }
            
            # Save scaler and metadata
            os.makedirs("models", exist_ok=True)
            with open("models/scaler.pkl", "wb") as f:
                pickle.dump(scaler, f)
            with open("models/dataset_info.json", "w", encoding="utf-8") as f:
                json.dump(dataset_metadata, f, indent=2, default=str)

            logger.info(f"Successfully loaded {len(X)} samples, split into {len(X_train)} train and {len(X_val)} validation")
            return X_train, X_val, y_train, y_val
            
        except Exception as e:
            logger.error(f"Error loading small CSV data from {csv_path}: {str(e)}")
            raise
    
    def _load_large_csv_data(self, csv_path, model_type, validation_split):
        """Load large CSV data with chunking, sampling, and validation"""
        file_size_mb = os.path.getsize(csv_path) / (1024 * 1024)
        logger.info(f"Loading large dataset ({file_size_mb:.1f}MB) with chunking...")
        
        try:
            # First pass: determine dataset structure from first chunk
            chunk_size = 10000
            first_chunk = pd.read_csv(csv_path, chunksize=chunk_size, nrows=chunk_size)
            first_chunk = next(first_chunk) if hasattr(first_chunk, '__next__') else first_chunk
            
            # Validate structure from first chunk
            metadata = self.validator.validate_csv_structure(first_chunk, csv_path)
            
            # Sample data intelligently
            total_rows = 0
            sample_data = []
            max_samples = 100000  # Increased sample size for better training
            
            # Try different encodings if needed
            encoding = 'utf-8'
            try:
                chunk_iterator = pd.read_csv(csv_path, chunksize=chunk_size, encoding=encoding)
            except UnicodeDecodeError:
                logger.warning("UTF-8 encoding failed, trying latin-1")
                encoding = 'latin-1'
                chunk_iterator = pd.read_csv(csv_path, chunksize=chunk_size, encoding=encoding)
            
            # Sample from chunks
            sample_rate = min(0.1, max_samples / 1000000)  # Adaptive sampling rate
            for chunk in chunk_iterator:
                total_rows += len(chunk)
                # Sample proportionally
                if len(sample_data) < max_samples:
                    remaining = max_samples - len(sample_data)
                    chunk_sample_size = min(len(chunk), max(1, int(len(chunk) * sample_rate)), remaining)
                    if chunk_sample_size > 0:
                        sample_chunk = chunk.sample(chunk_sample_size, random_state=42)
                        sample_data.append(sample_chunk)
        
            # Combine samples
            if not sample_data:
                raise ValueError("No data could be sampled from the dataset")
            
            sample_df = pd.concat(sample_data, ignore_index=True)
            logger.info(f"Sampled {len(sample_df)} rows from {total_rows} total rows for training")
        
            # Prepare features
            target_col = metadata['target_column']
            df_clean = self.validator.prepare_features(sample_df, handle_missing='drop')
            
            # Process data
            X = df_clean.drop(columns=[target_col]).values
            y = df_clean[target_col].values

            # Encode labels if necessary
            if not np.issubdtype(y.dtype, np.number):
                self.label_encoder = LabelEncoder()
                y = self.label_encoder.fit_transform(y)
                with open("models/label_encoder.pkl", "wb") as f:
                    pickle.dump(self.label_encoder, f)

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Validate validation split
            if validation_split <= 0 or validation_split >= 1:
                logger.warning(f"Invalid validation_split {validation_split}, using 0.2")
                validation_split = 0.2

            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y, test_size=validation_split, random_state=42,
                stratify=y if metadata['task_type'] == 'classification' and len(np.unique(y)) < 100 else None
            )

            # Save scaler and dataset info
            os.makedirs("models", exist_ok=True)
            with open("models/scaler.pkl", "wb") as f:
                pickle.dump(scaler, f)
        
            dataset_info = {
                "total_rows": total_rows,
                "sampled_rows": len(sample_df),
                "chunk_size": chunk_size,
                "original_csv_path": csv_path,
                "encoding": encoding,
                "metadata": metadata
            }
            with open("models/dataset_info.json", "w", encoding="utf-8") as f:
                json.dump(dataset_info, f, indent=2, default=str)

            logger.info(f"Successfully loaded and processed large dataset")
            return X_train, X_val, y_train, y_val
            
        except Exception as e:
            logger.error(f"Error loading large CSV data from {csv_path}: {str(e)}")
            raise

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