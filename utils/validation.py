"""
Data Validation Utility - Comprehensive dataset validation and preprocessing
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
import os
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """Comprehensive data validation and preprocessing"""
    
    @staticmethod
    def validate_dataset_path(dataset_path: str) -> bool:
        """Validate that dataset file or folder exists and is readable"""
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        # Allow both files (CSV) and directories (image folders for CNN)
        if not (os.path.isfile(dataset_path) or os.path.isdir(dataset_path)):
            raise ValueError(f"Path is not a file or directory: {dataset_path}")
        
        if not os.access(dataset_path, os.R_OK):
            raise PermissionError(f"Cannot read dataset: {dataset_path}")
        
        return True
    
    @staticmethod
    def detect_dataset_type(dataset_path: str) -> str:
        """Detect dataset type (CSV, images, etc.)"""
        ext = os.path.splitext(dataset_path)[1].lower()
        
        if ext == '.csv':
            return 'csv'
        elif ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            return 'image'
        else:
            raise ValueError(f"Unsupported dataset format: {ext}")
    
    @staticmethod
    def validate_csv_structure(df: pd.DataFrame, dataset_path: str) -> Dict[str, Any]:
        """Validate CSV structure and return metadata"""
        if df.empty:
            raise ValueError(f"Dataset is empty: {dataset_path}")
        
        # Check minimum rows
        if len(df) < 10:
            logger.warning(f"Dataset has only {len(df)} rows, which may be insufficient for training")
        
        # Check minimum columns (at least 1 feature + 1 target)
        if len(df.columns) < 2:
            raise ValueError(f"Dataset must have at least 2 columns (features + target), found {len(df.columns)}")
        
        # Detect target column (assume last column, but verify)
        target_col = df.columns[-1]
        
        # Check for missing values
        missing_features = df.iloc[:, :-1].isnull().sum()
        missing_target = df[target_col].isnull().sum()
        
        if missing_target > 0:
            logger.warning(f"Target column has {missing_target} missing values. These rows will be dropped.")
        
        # Check data types
        feature_cols = df.columns[:-1]
        numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = df[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Determine task type
        target_dtype = df[target_col].dtype
        is_numeric_target = pd.api.types.is_numeric_dtype(target_dtype)
        
        # Count unique values in target
        unique_targets = df[target_col].nunique()
        
        if is_numeric_target:
            task_type = 'regression'
        elif unique_targets <= 2:
            task_type = 'classification'
            num_classes = unique_targets
        elif unique_targets <= 20:
            task_type = 'classification'
            num_classes = unique_targets
        else:
            # Could be classification with many classes or regression
            # Default to classification but warn
            task_type = 'classification'
            num_classes = unique_targets
            logger.warning(f"Target has {unique_targets} unique values. Consider if this should be regression.")
        
        metadata = {
            'num_rows': len(df),
            'num_features': len(feature_cols),
            'target_column': target_col,
            'task_type': task_type,
            'num_classes': num_classes if task_type == 'classification' else None,
            'numeric_features': numeric_features,
            'categorical_features': categorical_features,
            'missing_features': missing_features.to_dict(),
            'missing_target': int(missing_target),
            'unique_targets': unique_targets,
            'feature_names': feature_cols.tolist()
        }
        
        return metadata
    
    @staticmethod
    def infer_task_type_from_target(y: np.ndarray) -> Tuple[str, Optional[int]]:
        """Infer task type from target values"""
        unique_values = np.unique(y)
        num_unique = len(unique_values)
        
        # Check if target is numeric
        is_numeric = np.issubdtype(y.dtype, np.number)
        
        if is_numeric:
            # If few unique values relative to total samples, might be classification
            if num_unique <= 2:
                return 'classification', num_unique
            elif num_unique <= 20 and num_unique < len(y) * 0.1:
                return 'classification', num_unique
            else:
                return 'regression', None
        else:
            # Categorical data is always classification
            return 'classification', num_unique
    
    @staticmethod
    def validate_model_config(model_config: Dict[str, Any], dataset_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and auto-correct model configuration based on dataset"""
        config = model_config.copy()
        
        # Auto-detect task type if not set or mismatched
        detected_task = dataset_metadata.get('task_type')
        if not config.get('task_type') or config.get('task_type') != detected_task:
            logger.info(f"Auto-detecting task type: {detected_task}")
            config['task_type'] = detected_task
        
        # Auto-set num_classes for classification
        if detected_task == 'classification':
            num_classes = dataset_metadata.get('num_classes')
            if not config.get('num_classes') or config.get('num_classes') != num_classes:
                logger.info(f"Auto-setting num_classes: {num_classes}")
                config['num_classes'] = num_classes
        
        # Validate model type compatibility
        model_type = config.get('model_type', '').lower()
        if model_type == 'decision_tree':
            # Decision trees work with both classification and regression
            pass
        elif model_type in ['cnn', 'rnn']:
            # Neural networks need numeric features
            if dataset_metadata.get('categorical_features'):
                logger.warning(f"Model {model_type} may not handle categorical features well. Consider encoding.")
        
        return config
    
    @staticmethod
    def prepare_features(df: pd.DataFrame, handle_missing: str = 'drop') -> pd.DataFrame:
        """Prepare features by handling missing values"""
        df_clean = df.copy()
        
        # Handle missing values in features
        if handle_missing == 'drop':
            # Drop rows with any missing values
            df_clean = df_clean.dropna()
        elif handle_missing == 'mean':
            # Fill numeric with mean
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
            # Fill categorical with mode
            categorical_cols = df_clean.select_dtypes(exclude=[np.number]).columns
            for col in categorical_cols:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'unknown')
        elif handle_missing == 'median':
            # Fill numeric with median
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
        
        if len(df_clean) == 0:
            raise ValueError("After handling missing values, dataset became empty")
        
        return df_clean
    
    @staticmethod
    def check_data_quality(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Check data quality and return warnings/issues"""
        issues = []
        warnings = []
        
        # Ensure X is numeric before checking quality
        if not np.issubdtype(X.dtype, np.number):
            try:
                # Try to convert to numeric
                import pandas as pd
                X_df = pd.DataFrame(X) if X.ndim == 2 else pd.Series(X)
                X = X_df.apply(pd.to_numeric, errors='coerce').values
                if X.ndim == 2:
                    # Remove columns that couldn't be converted
                    valid_cols = ~np.isnan(X).all(axis=0)
                    X = X[:, valid_cols]
                if X.size == 0 or (X.ndim == 2 and X.shape[1] == 0):
                    issues.append("No valid numeric features found in data")
                    return {"issues": issues, "warnings": warnings}
            except Exception as e:
                issues.append(f"Cannot check data quality: features contain non-numeric data ({str(e)})")
                return {"issues": issues, "warnings": warnings}
        
        # Check for constant features (zero variance)
        if X.ndim == 2:
            try:
                feature_variances = np.var(X, axis=0)
                constant_features = np.where(feature_variances < 1e-10)[0]
                if len(constant_features) > 0:
                    warnings.append(f"Found {len(constant_features)} constant features (columns: {constant_features.tolist()})")
            except Exception as e:
                warnings.append(f"Could not calculate feature variances: {str(e)}")
        
        # Check for NaN or Inf values
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            issues.append("Features contain NaN or Inf values")
        
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            issues.append("Target contains NaN or Inf values")
        
        # Check for class imbalance in classification
        if len(np.unique(y)) > 1 and len(np.unique(y)) < len(y) * 0.1:
            unique, counts = np.unique(y, return_counts=True)
            max_ratio = np.max(counts) / np.min(counts)
            if max_ratio > 10:
                warnings.append(f"Significant class imbalance detected (ratio: {max_ratio:.2f})")
        
        # Check data shape
        if X.ndim == 1:
            warnings.append("Features are 1D, consider reshaping for neural networks")
        
        return {
            'issues': issues,
            'warnings': warnings,
            'n_samples': len(X),
            'n_features': X.shape[-1] if X.ndim >= 2 else 1,
            'n_classes': len(np.unique(y)) if len(np.unique(y)) < 100 else None
        }
