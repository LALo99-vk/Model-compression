"""
SMART UNIVERSAL NORMALIZER - CRASH-PROOF VERSION
Handles: Tabular, Image, Text datasets for Decision Tree, CNN, RNN
Rules: Auto-detect everything, never crash, always succeed if validation passed
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UniversalDatasetNormalizer:
    """
    SMART UNIVERSAL NORMALIZER
    - Auto-detects dataset type (tabular/image/text)
    - Auto-detects task type (regression/classification)
    - Model-specific preprocessing (DT/CNN/RNN)
    - Never crashes - auto-fixes all issues
    - HARD RULE: If validation passed, this MUST succeed
    """
    
    def __init__(self):
        self.schema_info = None
        
    def normalize_dataset(self, dataset_path: str, model_type: str, 
                         validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Main entry point - NEVER crashes
        Returns standardized X_train, X_val, X_test, y_train, y_val, y_test
        """
        try:
            logger.info("🔄 SMART UNIVERSAL NORMALIZATION: Starting...")
            
            # STEP 1: Auto-detect dataset type and schema
            schema = self._auto_detect_dataset_and_schema(dataset_path, model_type)
            logger.info(f"✅ Detected: {schema['dataset_type']} dataset, {schema['n_features']} features, {schema['n_classes']} unique values")
            
            # STEP 2: Auto-detect task type (regression vs classification)
            task_type = self._auto_detect_task_type(schema)
            schema['task_type'] = task_type
            logger.info(f"✅ Task type: {task_type.upper()}")
            
            # STEP 3: Load and preprocess data (model-specific)
            X, y = self._load_and_preprocess(dataset_path, schema, model_type)
            logger.info(f"✅ Loaded: X shape={X.shape}, y shape={y.shape}")
            
            # STEP 4: Safe train/val/test split (auto-handles stratification)
            X_train, X_val, X_test, y_train, y_val, y_test = self._safe_split(
                X, y, validation_split, task_type, schema
            )
            logger.info(f"✅ Split complete: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
            
            # STEP 5: Final cleanup and validation
            X_train, X_val, X_test, y_train, y_val, y_test = self._final_cleanup(
                X_train, X_val, X_test, y_train, y_val, y_test, model_type, task_type
            )
            
            logger.info("✅ SMART UNIVERSAL NORMALIZATION: Complete!")
            
            return {
                "status": "success",
                "X_train": X_train,
                "X_val": X_val,
                "X_test": X_test,
                "y_train": y_train,
                "y_val": y_val,
                "y_test": y_test,
                "num_classes": 1 if task_type == 'regression' else schema['n_classes'],
                "schema": schema
            }
            
        except Exception as e:
            logger.error(f"❌ Normalization error: {str(e)}")
            # NEVER crash - return error response
            return {
                "status": "error",
                "message": f"Normalization failed: {str(e)}",
                "errors": [str(e)],
                "guidance": self._get_error_guidance(str(e), model_type)
            }
    
    def _auto_detect_dataset_and_schema(self, dataset_path: str, model_type: str) -> Dict[str, Any]:
        """
        STEP 1: Auto-detect dataset type and schema
        """
        schema = {
            'dataset_type': None,
            'n_features': 0,
            'n_classes': 0,
            'feature_columns': [],
            'label_column': None,
            'has_missing': False,
            'categorical_features': [],
            'numeric_features': []
        }
        
        # Detect based on file type
        if dataset_path.endswith('.txt'):
            # Text files → RNN
            schema['dataset_type'] = 'text'
            schema.update(self._detect_text_schema(dataset_path))
        elif dataset_path.endswith('.csv') or dataset_path.endswith('.xlsx'):
            schema['dataset_type'] = 'tabular'
            schema.update(self._detect_tabular_schema(dataset_path))
        elif os.path.isdir(dataset_path):
            # Check if images or text
            files = list(Path(dataset_path).rglob('*.*'))
            image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
            if any(f.suffix.lower() in image_exts for f in files):
                schema['dataset_type'] = 'image'
                schema.update(self._detect_image_schema(dataset_path))
            else:
                schema['dataset_type'] = 'text'
                schema.update(self._detect_text_schema(dataset_path))
        else:
            # Fallback: assume tabular
            logger.warning(f"⚠️ Unknown format {dataset_path}, assuming tabular")
            schema['dataset_type'] = 'tabular'
            schema.update(self._detect_tabular_schema(dataset_path))
        
        return schema
    
    def _detect_tabular_schema(self, dataset_path: str) -> Dict[str, Any]:
        """Detect schema for tabular datasets"""
        try:
            # Try reading as CSV
            df = pd.read_csv(dataset_path)
        except:
            # Try Excel
            try:
                df = pd.read_excel(dataset_path)
            except:
                logger.error("❌ Could not read file as CSV or Excel")
                return {}
        
        # Auto-detect label column (last column by default)
        label_col = df.columns[-1]
        feature_cols = df.columns[:-1].tolist()
        
        # Check for standard label column names
        label_keywords = ['label', 'target', 'class', 'output', 'y', 'result']
        for col in df.columns:
            if any(keyword in col.lower() for keyword in label_keywords):
                label_col = col
                feature_cols = [c for c in df.columns if c != label_col]
                break
        
        # Count unique values in label
        unique_labels = df[label_col].nunique()
        
        # Detect categorical vs numeric features
        categorical = []
        numeric = []
        for col in feature_cols:
            if df[col].dtype == 'object' or df[col].nunique() < 10:
                categorical.append(col)
            else:
                numeric.append(col)
        
        return {
            'n_features': len(feature_cols),
            'n_classes': unique_labels,
            'feature_columns': feature_cols,
            'label_column': label_col,
            'has_missing': df.isnull().any().any(),
            'categorical_features': categorical,
            'numeric_features': numeric
        }
    
    def _detect_image_schema(self, dataset_path: str) -> Dict[str, Any]:
        """Detect schema for image datasets"""
        # Count classes (subdirectories)
        subdirs = [d for d in Path(dataset_path).iterdir() if d.is_dir()]
        n_classes = len(subdirs) if subdirs else 1
        
        # Count images
        image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
        images = [f for f in Path(dataset_path).rglob('*.*') if f.suffix.lower() in image_exts]
        
        return {
            'n_features': 3 * 32 * 32,  # Assuming 32x32 RGB after resize
            'n_classes': n_classes,
            'feature_columns': ['image_data'],
            'label_column': 'folder_name',
            'has_missing': False,
            'n_images': len(images)
        }
    
    def _detect_text_schema(self, dataset_path: str) -> Dict[str, Any]:
        """Detect schema for text datasets (supports .txt and .csv files)"""
        
        # Check if it's a .txt file
        if dataset_path.lower().endswith('.txt'):
            try:
                with open(dataset_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                
                # Check if tab-separated (has labels)
                has_labels = any('\t' in line for line in lines[:10])
                
                if has_labels:
                    # Labeled text classification
                    labels = [line.split('\t')[0] for line in lines if '\t' in line]
                    n_classes = len(set(labels))
                    n_samples = len(lines)
                    seq_length = 50  # Word-based tokenization (reduced for speed)
                    logger.info(f"📄 Detected labeled text: {n_samples} samples, {n_classes} classes")
                else:
                    # Plain text → Character-level generation (OPTIMIZED params)
                    text_len = min(len(text), 100000)  # Limit text for speed
                    chars = sorted(list(set(text[:text_len])))
                    n_classes = len(chars)  # Each unique char is a class
                    seq_length = 50  # Reduced from 100 for faster training
                    step = 5  # Increased step for fewer sequences
                    n_samples = (text_len - seq_length) // step
                    logger.info(f"📄 Detected plain text: {text_len} chars, {n_classes} unique chars, ~{n_samples} sequences")
                
                return {
                    'n_features': seq_length,  # Sequence length
                    'n_classes': n_classes,
                    'text_column': 'text',
                    'label_column': 'label',
                    'n_samples': n_samples,
                    'is_char_level': not has_labels
                }
            except Exception as e:
                logger.warning(f"⚠️ Could not read .txt file: {e}")
                return {'n_features': 100, 'n_classes': 100}
        
        # For CSV files with text columns
        try:
            df = pd.read_csv(dataset_path)
            text_col = None
            label_col = df.columns[-1]
            
            # Find text column (longest strings)
            for col in df.columns[:-1]:
                if df[col].dtype == 'object':
                    avg_len = df[col].astype(str).str.len().mean()
                    if avg_len > 20:  # Likely text
                        text_col = col
                        break
            
            if not text_col:
                text_col = df.columns[0]
            
            unique_labels = df[label_col].nunique()
            
            return {
                'n_features': 100,  # Will be vocab size
                'n_classes': unique_labels,
                'feature_columns': [text_col],
                'label_column': label_col,
                'has_missing': df.isnull().any().any(),
                'text_column': text_col
            }
        except:
            return {
                'n_features': 100,
                'n_classes': 2,
                'feature_columns': ['text'],
                'label_column': 'label',
                'has_missing': False
            }
    
    def _auto_detect_task_type(self, schema: Dict[str, Any]) -> str:
        """
        STEP 2: Auto-detect task type
        RULE: Smart detection based on unique values and data characteristics
        """
        n_classes = schema['n_classes']
        dataset_type = schema.get('dataset_type', 'tabular')
        
        # SPECIAL CASE: Text data is always classification (predicting next char/word)
        if dataset_type == 'text':
            logger.info(f"💡 Text dataset → CLASSIFICATION task ({n_classes} classes/chars)")
            return 'classification'
        
        # SMART RULES for task type detection:
        # 1. Only 1 unique value → impossible, default to regression
        if n_classes <= 1:
            logger.warning(f"⚠️ Only {n_classes} unique value(s) - defaulting to REGRESSION")
            return 'regression'
        
        # 2. 2 classes → Binary classification
        elif n_classes == 2:
            logger.info(f"💡 2 unique values → BINARY CLASSIFICATION")
            return 'classification'
        
        # 3. 3-30 classes → Multi-class classification
        elif n_classes <= 30:
            logger.info(f"💡 {n_classes} unique values → CLASSIFICATION task")
            return 'classification'
        
        # 4. >30 unique values → Likely regression (continuous)
        else:
            logger.info(f"💡 {n_classes} unique values (>30) → REGRESSION task (continuous target)")
            return 'regression'
    
    def _load_and_preprocess(self, dataset_path: str, schema: Dict[str, Any], 
                             model_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        STEP 3: Load and preprocess based on dataset type and model type
        """
        dataset_type = schema['dataset_type']
        
        if dataset_type == 'tabular':
            return self._load_tabular(dataset_path, schema, model_type)
        elif dataset_type == 'image':
            return self._load_images(dataset_path, schema, model_type)
        elif dataset_type == 'text':
            return self._load_text(dataset_path, schema, model_type)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    def _load_tabular(self, dataset_path: str, schema: Dict[str, Any], 
                     model_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess tabular data - ULTRA CRASH-PROOF"""
        try:
            df = pd.read_csv(dataset_path)
        except:
            try:
                df = pd.read_excel(dataset_path)
            except:
                raise ValueError(f"Cannot read file as CSV or Excel: {dataset_path}")
        
        label_col = schema['label_column']
        feature_cols = schema['feature_columns']
        
        # Ensure columns exist
        if label_col not in df.columns:
            logger.warning(f"⚠️ Label column '{label_col}' not found, using last column")
            label_col = df.columns[-1]
        
        # Filter only existing feature columns
        existing_feature_cols = [col for col in feature_cols if col in df.columns]
        if not existing_feature_cols:
            logger.warning("⚠️ No feature columns found, using all except last")
            existing_feature_cols = df.columns[:-1].tolist()
        
        X = df[existing_feature_cols].copy()
        y = df[label_col].copy()
        
        # Handle missing values in features (NEVER crash)
        if X.isnull().any().any():
            logger.warning("⚠️ Missing values in features - imputing...")
            for col in X.columns:
                if X[col].isnull().any():
                    if X[col].dtype in [np.number, 'float64', 'int64']:
                        # Numeric: fill with median (or 0 if all NaN)
                        median_val = X[col].median()
                        fill_val = median_val if not pd.isna(median_val) else 0
                        X[col] = X[col].fillna(fill_val)
                    else:
                        # Categorical: fill with mode or 'unknown'
                        mode_val = X[col].mode()
                        fill_val = mode_val.iloc[0] if len(mode_val) > 0 else 'unknown'
                        X[col] = X[col].fillna(fill_val)
        
        # Handle missing values in target
        if y.isnull().any():
            logger.warning("⚠️ Missing values in target - removing those rows")
            valid_mask = ~y.isnull()
            X = X[valid_mask]
            y = y[valid_mask]
            if len(y) == 0:
                raise ValueError("All target values are missing - cannot train")
        
        # Encode categorical features
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                logger.info(f"🔄 Encoding categorical column: {col}")
                try:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                except Exception as e:
                    logger.warning(f"⚠️ Encoding failed for {col}: {e} - using hash encoding")
                    X[col] = X[col].astype(str).apply(lambda x: hash(x) % 10000)
        
        # Convert to numpy - handle any conversion errors
        try:
            X = X.values.astype(np.float32)
        except:
            logger.warning("⚠️ Cannot convert to float32 directly - forcing conversion")
            X = pd.DataFrame(X).apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32)
        
        # Handle target variable - SMART DETECTION
        if y.dtype == 'object' or y.dtype.name == 'category':
            # Categorical target - encode
            logger.info("🔄 Encoding categorical target")
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))
        else:
            # Numeric target - check if it needs re-indexing for classification
            y = y.values
            unique_count = len(np.unique(y))
            
            if unique_count > 30:
                # Regression - keep as is
                y = y.astype(np.float32)
            else:
                # Classification - MUST be 0-indexed for PyTorch!
                unique_vals = np.unique(y)
                
                # Check if labels are already 0-indexed
                if unique_vals.min() != 0 or unique_vals.max() != (len(unique_vals) - 1):
                    # Not 0-indexed, need to re-map
                    logger.info(f"🔄 Re-indexing labels from {list(unique_vals)} to [0, {len(unique_vals)-1}]")
                    label_map = {old_val: new_val for new_val, old_val in enumerate(unique_vals)}
                    y = np.array([label_map[val] for val in y])
                
                y = y.astype(np.int64)  # Classification
        
        # Ensure all values are finite
        X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
        y = np.nan_to_num(y, nan=0.0, posinf=1e10, neginf=-1e10)
        
        logger.info(f"✅ Tabular data loaded: X shape={X.shape}, y shape={y.shape}, y dtype={y.dtype}")
        
        return X, y
    
    def _load_images(self, dataset_path: str, schema: Dict[str, Any], 
                    model_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess images - CRASH-PROOF (supports flat and nested structures)"""
        from PIL import Image
        
        X_list = []
        y_list = []
        
        dataset_path_obj = Path(dataset_path)
        
        # Get class folders (subdirectories)
        class_folders = sorted([d for d in dataset_path_obj.iterdir() if d.is_dir()])
        
        if class_folders:
            # NESTED STRUCTURE: subdirectories = classes
            logger.info(f"📁 Loading images from {len(class_folders)} class folders")
            for class_idx, class_folder in enumerate(class_folders):
                image_files = [f for f in class_folder.iterdir() 
                              if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
                
                for img_file in image_files:
                    try:
                        img = Image.open(img_file).convert('RGB')
                        img = img.resize((32, 32))
                        img_array = np.array(img) / 255.0
                        # Transpose from HWC (32, 32, 3) to CHW (3, 32, 32) for PyTorch
                        img_array = np.transpose(img_array, (2, 0, 1))
                        X_list.append(img_array)
                        y_list.append(class_idx)
                    except Exception as e:
                        logger.warning(f"⚠️ Skipping corrupt image: {img_file.name}")
                        continue
        else:
            # FLAT STRUCTURE: all images in root folder (single class or unlabeled)
            logger.info(f"📄 Loading images from flat folder (single class)")
            image_files = [f for f in dataset_path_obj.iterdir() 
                          if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
            
            for img_file in image_files:
                try:
                    img = Image.open(img_file).convert('RGB')
                    img = img.resize((32, 32))
                    img_array = np.array(img) / 255.0
                    # Transpose from HWC (32, 32, 3) to CHW (3, 32, 32) for PyTorch
                    img_array = np.transpose(img_array, (2, 0, 1))
                    X_list.append(img_array)
                    y_list.append(0)  # Single class label
                except Exception as e:
                    logger.warning(f"⚠️ Skipping corrupt image: {img_file.name}")
                    continue
        
        if len(X_list) == 0:
            raise ValueError("No valid images found in dataset folder")
        
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int64)
        
        logger.info(f"✅ Loaded {len(X)} images, {len(set(y_list))} classes")
        
        return X, y
    
    def _load_text(self, dataset_path: str, schema: Dict[str, Any], 
                  model_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess text - CRASH-PROOF (supports .txt and .csv files)"""
        
        # Check if it's a .txt file or .csv file
        if dataset_path.lower().endswith('.txt'):
            logger.info("📄 Loading .txt file for RNN")
            with open(dataset_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            
            # Check if it's a labeled text file (tab-separated: label\ttext)
            lines = text.strip().split('\n')
            first_line = lines[0] if lines else ''
            
            if '\t' in first_line and len(first_line.split('\t')) == 2:
                # Tab-separated labeled text: label\ttext per line
                logger.info("📄 Detected labeled text format (label\\ttext)")
                texts = []
                labels = []
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split('\t', 1)
                    if len(parts) == 2:
                        labels.append(parts[0])
                        texts.append(parts[1])
                
                # Tokenize texts
                max_len = 100
                vocab_size = 10000
                tokenized = []
                for t in texts:
                    tokens = t.lower().split()[:max_len]
                    token_ids = [hash(tok) % vocab_size for tok in tokens]
                    token_ids = token_ids + [0] * (max_len - len(token_ids))
                    tokenized.append(token_ids)
                
                X = np.array(tokenized, dtype=np.float32)
                y = np.array(labels)
                
                if y.dtype == 'object':
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    y = le.fit_transform(y)
            else:
                # Plain text file → Character-level sequence prediction
                logger.info("📄 Detected plain text → Using character-level sequences for RNN")
                
                # OPTIMIZED: Limit text to prevent memory issues (100K chars max for faster processing)
                max_text_len = min(len(text), 100000)  # Reduced from 500K to 100K for speed
                text = text[:max_text_len]
                
                # Build character vocabulary from the LIMITED text
                chars = sorted(list(set(text)))
                char_to_idx = {c: i for i, c in enumerate(chars)}
                vocab_size = len(chars)
                
                logger.info(f"📊 Text length: {len(text)} chars, Vocab size: {vocab_size}")
                
                # Update schema with actual vocab size (CRITICAL for num_classes)
                schema['n_classes'] = vocab_size
                schema['vocab_size'] = vocab_size
                
                # OPTIMIZED: Create sequences using numpy for speed
                seq_length = 50  # Reduced from 100 for faster training
                step = 5  # Increased step for fewer sequences (faster)
                
                logger.info(f"📊 Creating sequences (seq_len={seq_length}, step={step})...")
                
                # Convert text to indices ONCE (vectorized)
                text_indices = np.array([char_to_idx[c] for c in text], dtype=np.int32)
                
                # Calculate number of sequences
                n_samples = (len(text_indices) - seq_length) // step
                
                if n_samples == 0:
                    raise ValueError(f"Text too short for sequence generation (need > {seq_length} chars)")
                
                # OPTIMIZED: Pre-allocate arrays instead of building lists
                X = np.zeros((n_samples, seq_length), dtype=np.float32)
                y = np.zeros(n_samples, dtype=np.int64)
                
                # Fill arrays using numpy slicing (much faster than Python loop)
                for i in range(n_samples):
                    start_idx = i * step
                    X[i] = text_indices[start_idx:start_idx + seq_length]
                    y[i] = text_indices[start_idx + seq_length]
                
                logger.info(f"📊 Created {n_samples} sequences of length {seq_length}")
                logger.info(f"📊 Labels range: 0 to {vocab_size - 1} (num_classes={vocab_size})")
                
                # Verify labels are within bounds
                max_label = y.max()
                if max_label >= vocab_size:
                    logger.error(f"❌ Label {max_label} >= vocab_size {vocab_size}")
                    raise ValueError(f"Label out of bounds: {max_label} >= {vocab_size}")
                
                # Save vocab for later use (generation)
                vocab_path = dataset_path.replace('.txt', '_vocab.json')
                try:
                    import json
                    with open(vocab_path, 'w') as f:
                        json.dump({
                            'char_to_idx': char_to_idx, 
                            'idx_to_char': {str(i): c for c, i in char_to_idx.items()},
                            'vocab_size': vocab_size
                        }, f)
                    logger.info(f"📁 Saved vocabulary to {vocab_path}")
                except Exception as e:
                    logger.warning(f"Could not save vocab: {e}")
                
                return X, y
            
        else:
            # Load CSV file (existing logic)
            logger.info("📄 Loading .csv file with text column")
            df = pd.read_csv(dataset_path)
            
            text_col = schema.get('text_column', df.columns[0])
            label_col = schema['label_column']
            
            texts = df[text_col].astype(str).values
            y = df[label_col].values
        
        # Simple tokenization and padding
        max_len = 100
        vocab_size = 10000
        
        # Basic tokenization (split by space)
        tokenized = []
        for text in texts:
            tokens = text.lower().split()[:max_len]
            # Convert to simple hash-based IDs
            token_ids = [hash(t) % vocab_size for t in tokens]
            # Pad to max_len
            token_ids = token_ids + [0] * (max_len - len(token_ids))
            tokenized.append(token_ids)
        
        X = np.array(tokenized, dtype=np.float32)
        
        # Encode labels if needed
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        return X, y
    
    def _safe_split(self, X: np.ndarray, y: np.ndarray, validation_split: float,
                   task_type: str, schema: Dict[str, Any]) -> Tuple:
        """
        STEP 4: ULTRA-SAFE train/val/test split - NEVER crashes
        RULE: Only stratify if classification AND all classes have ≥2 samples
        """
        # Determine if we can stratify
        can_stratify = False
        
        if task_type == 'classification':
            try:
                unique, counts = np.unique(y, return_counts=True)
                min_count = counts.min()
                
                # Additional check: need at least 3 samples per class for 3-way split
                if min_count >= 3:
                    can_stratify = True
                    logger.info(f"✅ Using stratified split (all classes have ≥{min_count} samples)")
                else:
                    logger.warning(f"⚠️ Cannot stratify: some classes have only {min_count} sample(s)")
                    logger.info("💡 Using regular split instead")
            except Exception as e:
                logger.warning(f"⚠️ Error checking class distribution: {e} - using regular split")
                can_stratify = False
        else:
            logger.info("💡 Regression task - using regular split")
        
        # Calculate split sizes - ensure we have enough data
        total_samples = len(X)
        if total_samples < 10:
            logger.warning(f"⚠️ Very small dataset ({total_samples} samples) - using simple 80/10/10 split")
            test_size = 0.1
            val_size = 0.1
        else:
            test_size = validation_split
            val_size = validation_split / 2
        
        try:
            # First split: train + (val+test)
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y,
                test_size=test_size + val_size,
                random_state=42,
                stratify=y if can_stratify else None
            )
            
            # Second split: val + test
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp,
                test_size=0.5,
                random_state=42,
                stratify=y_temp if can_stratify else None
            )
            
        except Exception as e:
            logger.warning(f"⚠️ Stratified split failed: {e}")
            logger.info("💡 Falling back to regular split (guaranteed to work)")
            
            try:
                # Fallback: split without stratification
                X_train, X_temp, y_train, y_temp = train_test_split(
                    X, y, test_size=test_size + val_size, random_state=42
                )
                X_val, X_test, y_val, y_test = train_test_split(
                    X_temp, y_temp, test_size=0.5, random_state=42
                )
            except Exception as e2:
                logger.error(f"❌ Even regular split failed: {e2}")
                # Last resort: manual split
                n = len(X)
                n_test = max(1, int(n * test_size))
                n_val = max(1, int(n * val_size))
                n_train = n - n_test - n_val
                
                X_train, y_train = X[:n_train], y[:n_train]
                X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
                X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]
                logger.info(f"✅ Manual split succeeded: Train={n_train}, Val={n_val}, Test={n_test}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _final_cleanup(self, X_train, X_val, X_test, y_train, y_val, y_test,
                      model_type: str, task_type: str) -> Tuple:
        """
        STEP 5: Final cleanup and validation
        Ensure all data is in correct format for each model type
        """
        # Ensure numpy arrays
        X_train = np.asarray(X_train, dtype=np.float32)
        X_val = np.asarray(X_val, dtype=np.float32)
        X_test = np.asarray(X_test, dtype=np.float32)
        
        # MODEL-SPECIFIC RESHAPING
        if model_type == 'rnn':
            # RNN expects 3D input: (batch, seq_len, input_size)
            if X_train.ndim == 2:
                logger.info("🔄 Reshaping 2D tabular data to 3D for RNN...")
                n_samples, n_features = X_train.shape
                
                # Auto-determine seq_len and input_size
                # Strategy: use seq_len=10 if possible, else adapt
                seq_len = min(10, n_features)
                input_size = max(1, n_features // seq_len)
                
                # Adjust if needed
                if n_features % seq_len != 0:
                    # Pad features to make it divisible
                    target_features = seq_len * input_size
                    if n_features < target_features:
                        pad_size = target_features - n_features
                        X_train = np.pad(X_train, ((0, 0), (0, pad_size)), mode='constant')
                        X_val = np.pad(X_val, ((0, 0), (0, pad_size)), mode='constant')
                        X_test = np.pad(X_test, ((0, 0), (0, pad_size)), mode='constant')
                    else:
                        X_train = X_train[:, :target_features]
                        X_val = X_val[:, :target_features]
                        X_test = X_test[:, :target_features]
                
                # Reshape to 3D
                X_train = X_train.reshape(-1, seq_len, input_size)
                X_val = X_val.reshape(-1, seq_len, input_size)
                X_test = X_test.reshape(-1, seq_len, input_size)
                
                logger.info(f"✅ Reshaped to 3D: {X_train.shape} (seq_len={seq_len}, input_size={input_size})")
        
        # Target variable dtype
        if task_type == 'regression':
            y_train = np.asarray(y_train, dtype=np.float32)
            y_val = np.asarray(y_val, dtype=np.float32)
            y_test = np.asarray(y_test, dtype=np.float32)
        else:
            y_train = np.asarray(y_train, dtype=np.int64)
            y_val = np.asarray(y_val, dtype=np.int64)
            y_test = np.asarray(y_test, dtype=np.int64)
        
        # Remove any NaN or Inf values (NEVER crash)
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e10, neginf=-1e10)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=1e10, neginf=-1e10)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e10, neginf=-1e10)
        y_train = np.nan_to_num(y_train, nan=0.0)
        y_val = np.nan_to_num(y_val, nan=0.0)
        y_test = np.nan_to_num(y_test, nan=0.0)
        
        logger.info(f"✅ Final shapes: X_train={X_train.shape}, y_train={y_train.shape}")
        logger.info(f"✅ Data types: X={X_train.dtype}, y={y_train.dtype}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _get_error_guidance(self, error: str, model_type: str) -> str:
        """Provide helpful error guidance"""
        guidance = f"Error during normalization: {error}\n\n"
        
        if model_type == 'decision_tree':
            guidance += "For Decision Tree models, ensure:\n"
            guidance += "- Dataset has numeric or categorical features\n"
            guidance += "- Target column is clearly identifiable\n"
            guidance += "- Missing values can be handled\n"
        elif model_type == 'cnn':
            guidance += "For CNN models, ensure:\n"
            guidance += "- Images are in proper folder structure\n"
            guidance += "- Image files are valid and not corrupt\n"
            guidance += "- All images can be read\n"
        elif model_type == 'rnn':
            guidance += "For RNN models, ensure:\n"
            guidance += "- Text data is in CSV format\n"
            guidance += "- Text column contains string data\n"
            guidance += "- Labels are clearly defined\n"
        
        return guidance
