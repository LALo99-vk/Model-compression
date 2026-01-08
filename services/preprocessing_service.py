"""
Preprocessing Service - Handles automatic preprocessing for all model types
Applies correct preprocessing, detects class imbalance, and provides detailed error messages
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from PIL import Image
import os
import pickle
import json
import re
import logging
from typing import Tuple, Dict, Any, Optional, List
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PreprocessingService:
    """Service for automatic preprocessing of datasets before training"""
    
    def __init__(self):
        self.scaler = None
        self.label_encoder = None
        self.preprocessing_warnings = []
        self.preprocessing_errors = []
        self.preprocessing_info = {}
        self.last_dataset_path = None  # Track last dataset to detect changes
        # Load existing scaler/encoder if available
        self._load_existing_artifacts()
        
    def preprocess_data(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        model_type: str,
        model_config: Dict[str, Any],
        dataset_path: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Apply model-specific preprocessing to training and validation data
        
        Returns:
            Tuple of (X_train_processed, X_val_processed, y_train_processed, y_val_processed, preprocessing_metadata)
        """
        self.preprocessing_warnings = []
        self.preprocessing_errors = []
        self.preprocessing_info = {}
        
        # Reset scaler if dataset path has changed (new dataset)
        if self.last_dataset_path is not None and self.last_dataset_path != dataset_path:
            logger.info(f"Dataset path changed from {self.last_dataset_path} to {dataset_path}, resetting scaler")
            self.scaler = None
        
        self.last_dataset_path = dataset_path
        
        try:
            logger.info(f"Starting preprocessing for {model_type} model...")
            
            if model_type == 'decision_tree':
                return self._preprocess_decision_tree(X_train, X_val, y_train, y_val, model_config)
            elif model_type == 'cnn':
                return self._preprocess_cnn(X_train, X_val, y_train, y_val, model_config)
            elif model_type == 'rnn':
                return self._preprocess_rnn(X_train, X_val, y_train, y_val, model_config, dataset_path)
            else:
                raise ValueError(f"Unsupported model type for preprocessing: {model_type}")
                
        except Exception as e:
            error_msg = f"Preprocessing failed: {str(e)}"
            fix_instructions = self._get_fix_instructions(model_type, str(e))
            full_error = f"{error_msg}\n\nHow to fix:\n{fix_instructions}"
            
            self.preprocessing_errors.append(full_error)
            logger.error(full_error, exc_info=True)
            
            # Save error details for user
            self._save_preprocessing_report()
            
            raise ValueError(full_error)
    
    def _preprocess_decision_tree(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        model_config: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Preprocess data for Decision Tree models"""
        try:
            logger.info("Applying Decision Tree preprocessing...")
            
            # 1. Handle missing values
            X_train, y_train = self._handle_missing_values(X_train, y_train, 'decision_tree')
            X_val, y_val = self._handle_missing_values(X_val, y_val, 'decision_tree')
            
            # 2. Ensure numeric features (should already be done, but double-check)
            if not np.issubdtype(X_train.dtype, np.number):
                logger.warning("Non-numeric features detected, attempting conversion...")
                X_train = pd.DataFrame(X_train).apply(pd.to_numeric, errors='coerce').values
                X_val = pd.DataFrame(X_val).apply(pd.to_numeric, errors='coerce').values
                
                # Check for columns that couldn't be converted
                nan_cols_train = np.isnan(X_train).all(axis=0)
                nan_cols_val = np.isnan(X_val).all(axis=0)
                
                if nan_cols_train.any() or nan_cols_val.any():
                    self.preprocessing_warnings.append(
                        f"Removed {nan_cols_train.sum() + nan_cols_val.sum()} non-numeric columns that couldn't be converted"
                    )
                    X_train = X_train[:, ~nan_cols_train]
                    X_val = X_val[:, ~nan_cols_val]
            
            # 3. Scale features (optional for Decision Tree, but helpful for consistency)
            # Check if data is already scaled (mean close to 0, std close to 1)
            if X_train.size > 0:
                # Get current number of features
                n_features = X_train.shape[1] if X_train.ndim > 1 else 1
                
                # Check if existing scaler is compatible (same number of features)
                scaler_compatible = False
                if self.scaler is not None:
                    # Check if scaler has been fitted and has the same number of features
                    if hasattr(self.scaler, 'n_features_in_') and self.scaler.n_features_in_ == n_features:
                        scaler_compatible = True
                    elif hasattr(self.scaler, 'mean_') and self.scaler.mean_.shape[0] == n_features:
                        scaler_compatible = True
                
                mean_check = np.abs(np.mean(X_train)) < 1e-6
                std_check = np.abs(np.std(X_train) - 1.0) < 0.1
                
                if mean_check and std_check and scaler_compatible:
                    # Data appears already scaled, use existing scaler
                    logger.info("Data appears already scaled, using existing scaler")
                    X_train_scaled = X_train
                    X_val_scaled = X_val
                elif not scaler_compatible:
                    # Feature count mismatch or no scaler - create new one
                    logger.info(f"Creating new scaler (feature count: {n_features}, existing scaler incompatible)")
                    self.scaler = StandardScaler()
                    X_train_scaled = self.scaler.fit_transform(X_train)
                    X_val_scaled = self.scaler.transform(X_val)
                else:
                    # Use existing compatible scaler
                    X_train_scaled = self.scaler.transform(X_train)
                    X_val_scaled = self.scaler.transform(X_val)
            else:
                X_train_scaled = X_train
                X_val_scaled = X_val
            
            # 4. Encode labels if needed
            y_train_encoded, y_val_encoded = self._encode_labels(y_train, y_val)
            
            # 5. Detect class imbalance
            imbalance_info = self._detect_class_imbalance(y_train_encoded, 'decision_tree')
            if imbalance_info:
                self.preprocessing_warnings.append(imbalance_info)
            
            # Save preprocessing artifacts
            self._save_preprocessing_artifacts()
            
            metadata = {
                'preprocessing_steps': ['missing_value_handling', 'numeric_conversion', 'feature_scaling', 'label_encoding'],
                'warnings': self.preprocessing_warnings,
                'scaler_saved': True,
                'label_encoder_saved': self.label_encoder is not None
            }
            
            logger.info("Decision Tree preprocessing completed successfully")
            return X_train_scaled, X_val_scaled, y_train_encoded, y_val_encoded, metadata
            
        except Exception as e:
            raise ValueError(f"Decision Tree preprocessing failed: {str(e)}")
    
    def _preprocess_cnn(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        model_config: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Preprocess data for CNN models (images)"""
        try:
            logger.info("Applying CNN preprocessing...")
            
            # CNN data should already be loaded as images, but we'll ensure proper format
            # 1. Verify image format
            if X_train.ndim != 4:
                raise ValueError(
                    f"CNN expects 4D image data (batch, channels, height, width), got {X_train.ndim}D. "
                    f"Please ensure images are properly loaded as numpy arrays."
                )
            
            # 2. Ensure images are normalized (0-1 range)
            if X_train.max() > 1.0:
                logger.info("Normalizing images to 0-1 range...")
                X_train = X_train / 255.0
                X_val = X_val / 255.0
                self.preprocessing_info['normalized'] = True
            else:
                self.preprocessing_info['normalized'] = False
                logger.info("Images already normalized")
            
            # 3. Verify image shape consistency
            expected_shape = X_train.shape[1:]  # (C, H, W)
            if X_val.shape[1:] != expected_shape:
                raise ValueError(
                    f"Image shape mismatch: Training images shape {expected_shape}, "
                    f"Validation images shape {X_val.shape[1:]}. All images must have the same dimensions."
                )
            
            # 4. Encode labels if needed
            y_train_encoded, y_val_encoded = self._encode_labels(y_train, y_val)
            
            # 5. Detect class imbalance
            imbalance_info = self._detect_class_imbalance(y_train_encoded, 'cnn')
            if imbalance_info:
                self.preprocessing_warnings.append(imbalance_info)
            
            # Save preprocessing artifacts
            self._save_preprocessing_artifacts()
            
            metadata = {
                'preprocessing_steps': ['normalization', 'shape_verification', 'label_encoding'],
                'image_shape': expected_shape,
                'warnings': self.preprocessing_warnings,
                'label_encoder_saved': self.label_encoder is not None
            }
            
            logger.info(f"CNN preprocessing completed successfully. Image shape: {expected_shape}")
            return X_train, X_val, y_train_encoded, y_val_encoded, metadata
            
        except Exception as e:
            raise ValueError(f"CNN preprocessing failed: {str(e)}")
    
    def _preprocess_rnn(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        model_config: Dict[str, Any],
        dataset_path: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Preprocess data for RNN models (sequences/text)"""
        try:
            logger.info("Applying RNN preprocessing...")
            
            # RNN requires sequences, so we need to prepare the data properly
            # Check if data is already in sequence format (3D: batch, seq_len, features)
            if X_train.ndim == 2:
                # Need to reshape to sequences
                logger.info("Reshaping 2D data to sequences for RNN...")
                seq_length = model_config.get('sequence_length', 10)
                
                # If we have text data (stored as file paths or as strings), we need to load and tokenize
                if dataset_path and os.path.exists(dataset_path):
                    # Check if dataset is text-based
                    try:
                        df = pd.read_csv(dataset_path, nrows=5)
                        text_cols = df.select_dtypes(include=['object']).columns
                        
                        if len(text_cols) > 0:
                            # We have text data - need to load full dataset and tokenize
                            logger.info("Detected text data, applying tokenization...")
                            return self._preprocess_rnn_text_data(dataset_path, model_config)
                    except Exception:
                        pass
                
                # For numeric tabular data, reshape to sequences
                # IMPORTANT: Reshaping reduces number of samples, so we need to adjust y accordingly
                X_train_seq, y_train_adjusted = self._reshape_to_sequences_with_labels(X_train, y_train, seq_length)
                X_val_seq, y_val_adjusted = self._reshape_to_sequences_with_labels(X_val, y_val, seq_length)
                y_train = y_train_adjusted
                y_val = y_val_adjusted
                
            elif X_train.ndim == 3:
                # Already in sequence format
                logger.info("Data already in sequence format")
                X_train_seq = X_train
                X_val_seq = X_val
                # No need to adjust y if already in sequence format
            else:
                raise ValueError(
                    f"RNN expects 2D (tabular) or 3D (sequence) data, got {X_train.ndim}D. "
                    f"Please ensure data is properly formatted for sequence learning."
                )
            
            # Verify shapes match before proceeding
            if len(X_train_seq) != len(y_train):
                raise ValueError(
                    f"Size mismatch: X_train has {len(X_train_seq)} sequences but y_train has {len(y_train)} labels. "
                    f"Please check preprocessing steps."
                )
            if len(X_val_seq) != len(y_val):
                raise ValueError(
                    f"Size mismatch: X_val has {len(X_val_seq)} sequences but y_val has {len(y_val)} labels. "
                    f"Please check preprocessing steps."
                )
            
            # Normalize features
            # Get feature dimension (last dimension of sequence data)
            n_features = X_train_seq.shape[-1] if X_train_seq.ndim > 1 else 1
            
            # Check if existing scaler is compatible (same number of features)
            scaler_compatible = False
            if self.scaler is not None:
                # Check if scaler has been fitted and has the same number of features
                if hasattr(self.scaler, 'n_features_in_') and self.scaler.n_features_in_ == n_features:
                    scaler_compatible = True
                elif hasattr(self.scaler, 'mean_') and self.scaler.mean_.shape[0] == n_features:
                    scaler_compatible = True
            
            if not scaler_compatible:
                # Create new scaler (either doesn't exist or incompatible)
                logger.info(f"Creating new scaler for RNN (feature count: {n_features}, existing scaler incompatible)")
                # Flatten for scaling, then reshape back
                original_shape = X_train_seq.shape
                X_train_flat = X_train_seq.reshape(-1, X_train_seq.shape[-1])
                self.scaler = StandardScaler()
                X_train_scaled_flat = self.scaler.fit_transform(X_train_flat)
                X_train_seq = X_train_scaled_flat.reshape(original_shape)
                
                X_val_flat = X_val_seq.reshape(-1, X_val_seq.shape[-1])
                X_val_scaled_flat = self.scaler.transform(X_val_flat)
                X_val_seq = X_val_scaled_flat.reshape(X_val_seq.shape)
            else:
                # Apply existing compatible scaler
                logger.info("Using existing compatible scaler for RNN")
                original_shape = X_train_seq.shape
                X_train_flat = X_train_seq.reshape(-1, X_train_seq.shape[-1])
                X_train_scaled_flat = self.scaler.transform(X_train_flat)
                X_train_seq = X_train_scaled_flat.reshape(original_shape)
                
                X_val_flat = X_val_seq.reshape(-1, X_val_seq.shape[-1])
                X_val_scaled_flat = self.scaler.transform(X_val_flat)
                X_val_seq = X_val_scaled_flat.reshape(X_val_seq.shape)
            
            # Encode labels
            y_train_encoded, y_val_encoded = self._encode_labels(y_train, y_val)
            
            # Detect class imbalance
            imbalance_info = self._detect_class_imbalance(y_train_encoded, 'rnn')
            if imbalance_info:
                self.preprocessing_warnings.append(imbalance_info)
            
            # Save preprocessing artifacts
            self._save_preprocessing_artifacts()
            
            metadata = {
                'preprocessing_steps': ['sequence_reshaping', 'feature_normalization', 'label_encoding'],
                'sequence_shape': X_train_seq.shape,
                'warnings': self.preprocessing_warnings,
                'scaler_saved': True,
                'label_encoder_saved': self.label_encoder is not None
            }
            
            logger.info(f"RNN preprocessing completed successfully. Sequence shape: {X_train_seq.shape}")
            return X_train_seq, X_val_seq, y_train_encoded, y_val_encoded, metadata
            
        except Exception as e:
            raise ValueError(f"RNN preprocessing failed: {str(e)}")
    
    def _preprocess_rnn_text_data(
        self,
        dataset_path: str,
        model_config: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Preprocess text data for RNN models with tokenization"""
        try:
            logger.info("Loading and tokenizing text data...")
            
            # Load CSV
            df = pd.read_csv(dataset_path)
            
            # Find text columns
            text_cols = df.select_dtypes(include=['object']).columns.tolist()
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not text_cols:
                raise ValueError(
                    "No text columns found in dataset. RNN text model requires text data. "
                    "Please ensure your dataset contains text columns."
                )
            
            # Use first text column as input
            text_col = text_cols[0]
            target_col = numeric_cols[-1] if numeric_cols else df.columns[-1]
            
            if target_col in text_cols:
                # If target is also text, use next text col
                text_col = text_cols[0] if text_cols[0] != target_col else (text_cols[1] if len(text_cols) > 1 else text_cols[0])
            
            logger.info(f"Using column '{text_col}' for text sequences and '{target_col}' as target")
            
            # Clean and tokenize text
            texts = df[text_col].astype(str).tolist()
            texts_cleaned = [self._clean_text(text) for text in texts]
            
            # Tokenize and create sequences
            seq_length = model_config.get('sequence_length', 50)
            max_vocab_size = model_config.get('vocab_size', 5000)
            
            X_seq, vocab_size = self._tokenize_and_pad_sequences(
                texts_cleaned, seq_length, max_vocab_size
            )
            
            # Get labels
            y = df[target_col].values
            
            # Split data
            validation_split = model_config.get('validation_split', 0.2)
            X_train, X_val, y_train, y_val = train_test_split(
                X_seq, y, test_size=validation_split, random_state=42,
                stratify=y if len(np.unique(y)) < 100 and len(np.unique(y)) > 1 else None
            )
            
            # Encode labels
            y_train_encoded, y_val_encoded = self._encode_labels(y_train, y_val)
            
            # Detect class imbalance
            imbalance_info = self._detect_class_imbalance(y_train_encoded, 'rnn')
            if imbalance_info:
                self.preprocessing_warnings.append(imbalance_info)
            
            metadata = {
                'preprocessing_steps': ['text_cleaning', 'tokenization', 'sequence_padding', 'label_encoding'],
                'sequence_shape': X_train.shape,
                'vocab_size': vocab_size,
                'text_column': text_col,
                'target_column': target_col,
                'warnings': self.preprocessing_warnings
            }
            
            logger.info(f"RNN text preprocessing completed. Sequence shape: {X_train.shape}, Vocab size: {vocab_size}")
            return X_train, X_val, y_train_encoded, y_val_encoded, metadata
            
        except Exception as e:
            raise ValueError(f"RNN text preprocessing failed: {str(e)}")
    
    def _clean_text(self, text: str) -> str:
        """Clean text data"""
        if pd.isna(text) or text == '':
            return ''
        
        # Remove special characters but keep spaces and basic punctuation
        text = re.sub(r'[^\w\s.,!?]', '', str(text))
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text.lower()
    
    def _tokenize_and_pad_sequences(
        self,
        texts: List[str],
        seq_length: int,
        max_vocab_size: int
    ) -> Tuple[np.ndarray, int]:
        """Tokenize text and pad/truncate sequences"""
        # Simple word-level tokenization
        word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        word_counts = Counter()
        
        # Count words
        for text in texts:
            words = text.split()
            word_counts.update(words)
        
        # Create vocabulary (keep most common words)
        most_common = word_counts.most_common(max_vocab_size - 2)  # -2 for PAD and UNK
        for word, count in most_common:
            if word not in word_to_idx:
                word_to_idx[word] = len(word_to_idx)
        
        vocab_size = len(word_to_idx)
        
        # Tokenize texts
        sequences = []
        for text in texts:
            words = text.split()
            tokens = [word_to_idx.get(word, word_to_idx['<UNK>']) for word in words]
            
            # Pad or truncate
            if len(tokens) < seq_length:
                tokens = tokens + [word_to_idx['<PAD>']] * (seq_length - len(tokens))
            else:
                tokens = tokens[:seq_length]
            
            sequences.append(tokens)
        
        # Convert to numpy array and reshape for RNN (batch, seq_len, 1)
        X = np.array(sequences, dtype=np.int32)
        X = X.reshape(X.shape[0], X.shape[1], 1)  # Add feature dimension
        
        # Save vocabulary for later use
        os.makedirs("models", exist_ok=True)
        with open("models/rnn_vocab.json", "w", encoding="utf-8") as f:
            json.dump(word_to_idx, f, indent=2)
        
        return X, vocab_size
    
    def _reshape_to_sequences(self, X: np.ndarray, seq_length: int) -> np.ndarray:
        """Reshape tabular data to sequences"""
        if X.ndim != 2:
            raise ValueError(f"Expected 2D data for sequence reshaping, got {X.ndim}D")
        
        n_samples, n_features = X.shape
        
        # Create sequences by sliding window
        sequences = []
        for i in range(n_samples - seq_length + 1):
            seq = X[i:i+seq_length]
            sequences.append(seq)
        
        if len(sequences) == 0:
            # If not enough samples, pad with zeros
            pad_seq = np.zeros((seq_length, n_features))
            pad_seq[:n_samples] = X
            sequences = [pad_seq]
        
        X_seq = np.array(sequences)
        
        # Ensure shape is (batch, seq_len, features)
        if X_seq.ndim == 2:
            X_seq = X_seq.reshape(X_seq.shape[0], X_seq.shape[1], 1)
        
        return X_seq
    
    def _reshape_to_sequences_with_labels(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        seq_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Reshape tabular data to sequences and adjust labels accordingly"""
        if X.ndim != 2:
            raise ValueError(f"Expected 2D data for sequence reshaping, got {X.ndim}D")
        
        n_samples, n_features = X.shape
        
        # Verify X and y have same length
        if len(X) != len(y):
            raise ValueError(
                f"X and y must have same length. Got X: {len(X)}, y: {len(y)}"
            )
        
        # Create sequences by sliding window
        # For each sequence, use the label of the LAST sample in the sequence
        sequences = []
        labels = []
        
        for i in range(n_samples - seq_length + 1):
            seq = X[i:i+seq_length]
            sequences.append(seq)
            # Use label of the last sample in the sequence
            labels.append(y[i + seq_length - 1])
        
        if len(sequences) == 0:
            # If not enough samples, pad with zeros and use the last label
            pad_seq = np.zeros((seq_length, n_features))
            pad_seq[:n_samples] = X
            sequences = [pad_seq]
            labels = [y[-1]] if len(y) > 0 else [0]
        
        X_seq = np.array(sequences)
        y_adjusted = np.array(labels)
        
        # Ensure shape is (batch, seq_len, features)
        if X_seq.ndim == 2:
            X_seq = X_seq.reshape(X_seq.shape[0], X_seq.shape[1], 1)
        
        logger.info(f"Reshaped {n_samples} samples to {len(X_seq)} sequences (seq_length={seq_length})")
        
        return X_seq, y_adjusted
    
    def _handle_missing_values(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Handle missing values in features and target"""
        # Check for missing values in target
        if np.isnan(y).any():
            missing_count = np.isnan(y).sum()
            self.preprocessing_warnings.append(
                f"Found {missing_count} missing values in target. Dropping these samples."
            )
            valid_mask = ~np.isnan(y)
            X = X[valid_mask]
            y = y[valid_mask]
        
        # Check for missing values in features
        if np.isnan(X).any():
            missing_per_col = np.isnan(X).sum(axis=0)
            cols_with_missing = np.where(missing_per_col > 0)[0]
            
            if len(cols_with_missing) > 0:
                self.preprocessing_warnings.append(
                    f"Found missing values in {len(cols_with_missing)} feature columns. "
                    f"Filling with median values."
                )
                
                # Fill with median for each column
                for col_idx in cols_with_missing:
                    col = X[:, col_idx]
                    median_val = np.nanmedian(col)
                    X[:, col_idx] = np.where(np.isnan(col), median_val, col)
        
        return X, y
    
    def _encode_labels(
        self,
        y_train: np.ndarray,
        y_val: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Encode labels if they are not numeric"""
        # Check if encoding is needed
        if np.issubdtype(y_train.dtype, np.number):
            return y_train, y_val
        
        # Encode labels
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            y_train_encoded = self.label_encoder.fit_transform(y_train.astype(str))
        else:
            # Use existing encoder, handling new labels
            try:
                y_train_encoded = self.label_encoder.transform(y_train.astype(str))
            except ValueError:
                # New labels found, refit encoder
                self.label_encoder = LabelEncoder()
                y_train_encoded = self.label_encoder.fit_transform(y_train.astype(str))
        
        # Encode validation labels
        try:
            y_val_encoded = self.label_encoder.transform(y_val.astype(str))
        except ValueError as e:
            logger.warning(f"Some validation labels not seen in training: {str(e)}")
            # For unseen labels, assign the most common class
            most_common_label = self.label_encoder.classes_[0]
            y_val_str = y_val.astype(str)
            y_val_encoded = np.array([
                self.label_encoder.transform([label])[0] if label in self.label_encoder.classes_ 
                else self.label_encoder.transform([most_common_label])[0]
                for label in y_val_str
            ])
        
        return y_train_encoded, y_val_encoded
    
    def _detect_class_imbalance(
        self,
        y: np.ndarray,
        model_type: str
    ) -> Optional[str]:
        """Detect and warn about class imbalance"""
        # Only check for classification tasks
        unique_labels = np.unique(y)
        
        if len(unique_labels) < 2:
            return None  # Not a classification problem or only one class
        
        if len(unique_labels) > 100:
            return None  # Likely regression
        
        # Count classes
        label_counts = Counter(y)
        counts = list(label_counts.values())
        max_count = max(counts)
        min_count = min(counts)
        ratio = max_count / min_count if min_count > 0 else float('inf')
        
        # Warn if ratio > 5
        if ratio > 5:
            class_distribution = {int(k): int(v) for k, v in label_counts.items()}
            warning_msg = (
                f"⚠️ CLASS IMBALANCE DETECTED: "
                f"Class ratio is {ratio:.2f}:1 (max: {max_count}, min: {min_count}). "
                f"Distribution: {class_distribution}. "
                f"This may affect model performance. Consider using class weights or resampling techniques."
            )
            
            self.preprocessing_info['class_imbalance'] = {
                'ratio': float(ratio),
                'max_count': int(max_count),
                'min_count': int(min_count),
                'distribution': class_distribution
            }
            
            return warning_msg
        
        return None
    
    def _get_fix_instructions(self, model_type: str, error_msg: str) -> str:
        """Generate fix instructions based on error message"""
        instructions = []
        
        if "missing values" in error_msg.lower():
            instructions.append("• Check your dataset for empty cells or NaN values")
            instructions.append("• Fill missing values manually or use the Auto-Fix feature")
        
        if "non-numeric" in error_msg.lower() or "categorical" in error_msg.lower():
            instructions.append("• Convert categorical columns to numeric using encoding")
            instructions.append("• Ensure all feature columns contain numeric data")
        
        if "shape" in error_msg.lower() or "dimension" in error_msg.lower():
            if model_type == 'cnn':
                instructions.append("• Ensure all images have the same dimensions")
                instructions.append("• Check that images are properly loaded")
            elif model_type == 'rnn':
                instructions.append("• Ensure data is properly formatted for sequences")
                instructions.append("• Check sequence length configuration")
        
        if "empty" in error_msg.lower():
            instructions.append("• Ensure your dataset has sufficient data (minimum 10 samples)")
            instructions.append("• Check that the dataset file is not corrupted")
        
        if not instructions:
            instructions.append("• Check the dataset format matches the selected model type")
            instructions.append("• Verify all required columns are present")
            instructions.append("• Try using the Auto-Fix Dataset feature")
        
        return "\n".join(instructions)
    
    def _load_existing_artifacts(self):
        """Load existing preprocessing artifacts if available"""
        # Load scaler if exists
        scaler_path = "models/scaler.pkl"
        if os.path.exists(scaler_path):
            try:
                with open(scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
                logger.info("Loaded existing scaler from models/scaler.pkl")
            except Exception as e:
                logger.warning(f"Could not load existing scaler: {str(e)}")
        
        # Load label encoder if exists
        encoder_path = "models/label_encoder.pkl"
        if os.path.exists(encoder_path):
            try:
                with open(encoder_path, "rb") as f:
                    self.label_encoder = pickle.load(f)
                logger.info("Loaded existing label encoder from models/label_encoder.pkl")
            except Exception as e:
                logger.warning(f"Could not load existing label encoder: {str(e)}")
    
    def _save_preprocessing_artifacts(self):
        """Save preprocessing artifacts (scalers, encoders)"""
        os.makedirs("models", exist_ok=True)
        
        if self.scaler is not None:
            with open("models/scaler.pkl", "wb") as f:
                pickle.dump(self.scaler, f)
            logger.info("Saved scaler to models/scaler.pkl")
        
        if self.label_encoder is not None:
            with open("models/label_encoder.pkl", "wb") as f:
                pickle.dump(self.label_encoder, f)
            logger.info("Saved label encoder to models/label_encoder.pkl")
    
    def _save_preprocessing_report(self):
        """Save preprocessing warnings and errors to a report file"""
        os.makedirs("results", exist_ok=True)
        
        report = {
            'warnings': self.preprocessing_warnings,
            'errors': self.preprocessing_errors,
            'info': self.preprocessing_info
        }
        
        with open("results/preprocessing_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        
        # Create human-readable report
        report_text = "📊 PREPROCESSING REPORT\n"
        report_text += "=" * 60 + "\n\n"
        
        if self.preprocessing_warnings:
            report_text += "⚠️  WARNINGS:\n"
            for i, warning in enumerate(self.preprocessing_warnings, 1):
                report_text += f"  {i}. {warning}\n"
            report_text += "\n"
        
        if self.preprocessing_errors:
            report_text += "❌ ERRORS:\n"
            for i, error in enumerate(self.preprocessing_errors, 1):
                report_text += f"  {i}. {error}\n"
            report_text += "\n"
        
        if self.preprocessing_info:
            report_text += "ℹ️  INFORMATION:\n"
            for key, value in self.preprocessing_info.items():
                report_text += f"  • {key}: {value}\n"
            report_text += "\n"
        
        if not self.preprocessing_warnings and not self.preprocessing_errors:
            report_text += "✅ No issues detected during preprocessing.\n"
        
        with open("results/preprocessing_report.txt", "w", encoding="utf-8") as f:
            f.write(report_text)
        
        logger.info("Saved preprocessing report to results/preprocessing_report.txt")
    
    def get_preprocessing_warnings(self) -> List[str]:
        """Get list of preprocessing warnings"""
        return self.preprocessing_warnings
    
    def get_preprocessing_errors(self) -> List[str]:
        """Get list of preprocessing errors"""
        return self.preprocessing_errors
    
    def get_preprocessing_info(self) -> Dict[str, Any]:
        """Get preprocessing information"""
        return self.preprocessing_info

