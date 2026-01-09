"""
Phase 1: Dataset Validation & Conditioning Service
Comprehensive validation before training to ensure dataset-model compatibility
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from PIL import Image
import json
import logging
import glob
from pathlib import Path

logger = logging.getLogger(__name__)


class DatasetValidationService:
    """Comprehensive dataset validation for ML workflow Phase 1"""
    
    def __init__(self):
        self.validation_reports = {}
    
    def validate_dataset(self, dataset_path: str, model_type: str) -> Dict[str, Any]:
        """
        Main validation method - validates dataset against model requirements
        
        Returns:
            {
                'is_valid': bool,
                'model_type': str,
                'dataset_path': str,
                'status': 'VALID' | 'INVALID',
                'issues': List[str],
                'warnings': List[str],
                'info': Dict[str, Any],
                'can_auto_fix': bool,
                'fix_suggestions': List[str]
            }
        """
        logger.info(f"🚦 PHASE 1: Validating dataset for {model_type} model")
        
        # Initialize report
        report = {
            'is_valid': True,
            'model_type': model_type.lower(),
            'dataset_path': dataset_path,
            'status': 'VALID',
            'issues': [],
            'warnings': [],
            'info': {},
            'can_auto_fix': False,
            'fix_suggestions': []
        }
        
        # 1.1 Basic file validation
        file_validation = self._validate_file_structure(dataset_path)
        if not file_validation['is_valid']:
            report['is_valid'] = False
            report['status'] = 'INVALID'
            report['issues'].extend(file_validation['issues'])
            return report
        
        report['info'].update(file_validation['info'])
        
        # 1.2 Model-specific validation
        if model_type.lower() == 'decision_tree':
            model_validation = self._validate_decision_tree_dataset(dataset_path)
        elif model_type.lower() == 'cnn':
            model_validation = self._validate_cnn_dataset(dataset_path)
        elif model_type.lower() == 'rnn':
            model_validation = self._validate_rnn_dataset(dataset_path)
        else:
            report['is_valid'] = False
            report['status'] = 'INVALID'
            report['issues'].append(f"Unknown model type: {model_type}")
            return report
        
        # Merge validation results
        if not model_validation['is_valid']:
            report['is_valid'] = False
            report['status'] = 'INVALID'
        
        report['issues'].extend(model_validation['issues'])
        report['warnings'].extend(model_validation['warnings'])
        report['info'].update(model_validation['info'])
        report['can_auto_fix'] = model_validation.get('can_auto_fix', False)
        report['fix_suggestions'] = model_validation.get('fix_suggestions', [])
        
        logger.info(f"Validation complete: {report['status']} ({len(report['issues'])} issues, {len(report['warnings'])} warnings)")
        
        return report
    
    def _validate_file_structure(self, dataset_path: str) -> Dict[str, Any]:
        """Validate basic file structure and accessibility"""
        issues = []
        info = {}
        
        # Check file exists
        if not os.path.exists(dataset_path):
            return {
                'is_valid': False,
                'issues': [f"Dataset file not found: {dataset_path}"],
                'info': {}
            }
        
        # Check file readability
        if not os.access(dataset_path, os.R_OK):
            return {
                'is_valid': False,
                'issues': [f"Cannot read dataset file (permission denied): {dataset_path}"],
                'info': {}
            }
        
        # Get file info
        file_size = os.path.getsize(dataset_path)
        file_ext = os.path.splitext(dataset_path)[1].lower()
        
        info['file_size'] = file_size
        info['file_size_mb'] = round(file_size / (1024 * 1024), 2)
        info['file_extension'] = file_ext
        info['file_path'] = dataset_path
        
        # Check if it's a directory (for image datasets)
        if os.path.isdir(dataset_path):
            info['is_directory'] = True
        else:
            info['is_directory'] = False
            # Validate file extension
            valid_extensions = ['.csv', '.txt', '.xlsx', '.xls', '.json', '.jpg', '.jpeg', '.png', '.bmp']
            if file_ext not in valid_extensions:
                issues.append(f"Unsupported file format: {file_ext}. Supported: {', '.join(valid_extensions)}")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'info': info
        }
    
    def _validate_decision_tree_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """Validate dataset for Decision Tree (tabular CSV/Excel format)"""
        issues = []
        warnings = []
        info = {}
        fix_suggestions = []
        
        try:
            # Check file extension
            ext = os.path.splitext(dataset_path)[1].lower()
            if ext not in ['.csv', '.xlsx', '.xls']:
                issues.append(f"Decision Tree requires CSV/Excel format, found: {ext}")
                fix_suggestions.append("Convert file to CSV format")
                return {
                    'is_valid': False,
                    'issues': issues,
                    'warnings': warnings,
                    'info': info,
                    'can_auto_fix': False,
                    'fix_suggestions': fix_suggestions
                }
            
            # Try to load CSV
            try:
                if ext == '.csv':
                    df = pd.read_csv(dataset_path, encoding='utf-8', nrows=1000)  # Sample first 1000 rows
                else:
                    # Try to read Excel (requires openpyxl)
                    try:
                        df = pd.read_excel(dataset_path, nrows=1000)
                    except ImportError:
                        issues.append("Excel file support requires 'openpyxl' package. Install with: pip install openpyxl")
                        return {
                            'is_valid': False,
                            'issues': issues,
                            'warnings': warnings,
                            'info': info,
                            'can_auto_fix': False,
                            'fix_suggestions': fix_suggestions
                        }
            except UnicodeDecodeError:
                try:
                    if ext == '.csv':
                        df = pd.read_csv(dataset_path, encoding='latin-1', nrows=1000)
                    else:
                        df = pd.read_excel(dataset_path, nrows=1000)
                except Exception as e:
                    issues.append(f"Cannot read file: {str(e)}")
                    return {
                        'is_valid': False,
                        'issues': issues,
                        'warnings': warnings,
                        'info': info,
                        'can_auto_fix': False,
                        'fix_suggestions': fix_suggestions
                    }
            except Exception as e:
                issues.append(f"Error reading file: {str(e)}")
                return {
                    'is_valid': False,
                    'issues': issues,
                    'warnings': warnings,
                    'info': info,
                    'can_auto_fix': False,
                    'fix_suggestions': fix_suggestions
                }
            
            # Check if empty
            if df.empty:
                issues.append("Dataset is empty (no rows found)")
                return {
                    'is_valid': False,
                    'issues': issues,
                    'warnings': warnings,
                    'info': info,
                    'can_auto_fix': False,
                    'fix_suggestions': fix_suggestions
                }
            
            # Check minimum rows
            total_rows = sum(1 for _ in open(dataset_path, encoding='utf-8')) - 1 if ext == '.csv' else len(df)
            if total_rows < 10:
                issues.append(f"Insufficient samples: {total_rows} rows. Minimum required: 10")
            
            # Check minimum columns (at least 1 feature + 1 target)
            if len(df.columns) < 2:
                issues.append(f"Dataset must have at least 2 columns (features + target), found: {len(df.columns)}")
                return {
                    'is_valid': False,
                    'issues': issues,
                    'warnings': warnings,
                    'info': info,
                    'can_auto_fix': False,
                    'fix_suggestions': fix_suggestions
                }
            
            # Check for missing values
            target_col = df.columns[-1]
            missing_target = df[target_col].isnull().sum()
            missing_features = df.iloc[:, :-1].isnull().sum().sum()
            
            if missing_target > 0:
                issues.append(f"Target column '{target_col}' has {missing_target} missing values")
                fix_suggestions.append("Drop rows with missing target values")
            
            if missing_features > 0:
                warnings.append(f"Feature columns have {missing_features} missing values")
                fix_suggestions.append("Fill or drop missing values in feature columns")
            
            # Check data types
            feature_cols = df.columns[:-1]
            numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
            categorical_features = df[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()
            
            if categorical_features:
                warnings.append(f"Found {len(categorical_features)} categorical feature columns")
                fix_suggestions.append("Encode categorical features to numeric")
            
            # Check for completely empty columns
            empty_cols = [col for col in df.columns if df[col].isnull().all()]
            if empty_cols:
                issues.append(f"Found {len(empty_cols)} completely empty columns: {empty_cols}")
                fix_suggestions.append("Remove empty columns")
            
            # Check target column consistency
            target_unique = df[target_col].nunique()
            if target_unique == 0:
                issues.append("Target column has no unique values")
            elif target_unique == 1:
                issues.append("Target column has only one unique value (cannot train model)")
            
            info.update({
                'num_rows': total_rows,
                'num_features': len(feature_cols),
                'num_numeric_features': len(numeric_features),
                'num_categorical_features': len(categorical_features),
                'target_column': target_col,
                'target_unique_values': target_unique,
                'missing_values_target': int(missing_target),
                'missing_values_features': int(missing_features),
                'empty_columns': empty_cols
            })
            
            can_auto_fix = bool(fix_suggestions) and len(issues) > 0
            
        except Exception as e:
            logger.error(f"Error validating Decision Tree dataset: {str(e)}", exc_info=True)
            issues.append(f"Validation error: {str(e)}")
            return {
                'is_valid': False,
                'issues': issues,
                'warnings': warnings,
                'info': info,
                'can_auto_fix': False,
                'fix_suggestions': fix_suggestions
            }
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'info': info,
            'can_auto_fix': can_auto_fix,
            'fix_suggestions': fix_suggestions
        }
    
    def _validate_cnn_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """Validate dataset for CNN (image folders with labeled directories)"""
        issues = []
        warnings = []
        info = {}
        fix_suggestions = []
        
        try:
            # Check if path is directory
            if not os.path.isdir(dataset_path):
                issues.append("CNN requires image directory structure, found file instead")
                fix_suggestions.append("Organize images into labeled subdirectories")
                return {
                    'is_valid': False,
                    'issues': issues,
                    'warnings': warnings,
                    'info': info,
                    'can_auto_fix': False,
                    'fix_suggestions': fix_suggestions
                }
            
            # Check for image files
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            image_files = []
            for ext in image_extensions:
                image_files.extend(glob.glob(os.path.join(dataset_path, '**', ext), recursive=True))
            
            if not image_files:
                issues.append("No image files found in directory (looking for .jpg, .jpeg, .png, .bmp)")
                return {
                    'is_valid': False,
                    'issues': issues,
                    'warnings': warnings,
                    'info': info,
                    'can_auto_fix': False,
                    'fix_suggestions': fix_suggestions
                }
            
            # Check minimum images
            if len(image_files) < 10:
                issues.append(f"Insufficient images: {len(image_files)} found. Minimum required: 10")
            
            # Check directory structure (subdirectories = labels)
            subdirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
            
            # Initialize images_per_class (always needed later)
            images_per_class = {}
            
            if not subdirs:
                # Images might be in root with labels in filenames
                warnings.append("No subdirectories found. Expected: subdirectories named by class labels")
                fix_suggestions.append("Organize images into subdirectories by class labels")
                info['structure'] = 'flat'
            else:
                info['structure'] = 'nested'
                info['label_directories'] = subdirs
                
                # Check images in each label directory
                for label_dir in subdirs:
                    label_path = os.path.join(dataset_path, label_dir)
                    label_images = []
                    for ext in image_extensions:
                        label_images.extend(glob.glob(os.path.join(label_path, ext)))
                    images_per_class[label_dir] = len(label_images)
                    
                    if len(label_images) == 0:
                        issues.append(f"Label directory '{label_dir}' contains no images")
                        fix_suggestions.append(f"Add images to '{label_dir}' directory or remove empty directory")
            
            # Check image consistency (size, format, corruption)
            checked_images = 0
            size_issues = []
            corrupted_images = []
            sample_sizes = []
            
            for img_path in image_files[:100]:  # Sample first 100 images
                try:
                    img = Image.open(img_path)
                    width, height = img.size
                    sample_sizes.append((width, height))
                    
                    if width < 32 or height < 32:
                        size_issues.append(f"{os.path.basename(img_path)}: too small ({width}x{height})")
                    
                    checked_images += 1
                except Exception as e:
                    corrupted_images.append(f"{os.path.basename(img_path)}: {str(e)}")
            
            if len(set(sample_sizes)) > 10:
                warnings.append(f"Images have inconsistent sizes (found {len(set(sample_sizes))} different sizes)")
                fix_suggestions.append("Resize all images to consistent size (e.g., 32x32 or 64x64)")
            
            if corrupted_images:
                issues.append(f"Found {len(corrupted_images)} corrupted/unreadable images")
                fix_suggestions.append("Remove or fix corrupted images")
            
            if size_issues:
                warnings.append(f"Found {len(size_issues)} images that are too small (< 32x32)")
                fix_suggestions.append("Resize small images to at least 32x32 pixels")
            
            info.update({
                'num_images': len(image_files),
                'num_classes': len(subdirs) if subdirs else 1,
                'images_per_class': images_per_class,
                'checked_images': checked_images,
                'corrupted_images_count': len(corrupted_images),
                'inconsistent_sizes': len(set(sample_sizes)) > 10
            })
            
            can_auto_fix = bool(fix_suggestions) and (corrupted_images or len(set(sample_sizes)) > 10)
            
        except Exception as e:
            logger.error(f"Error validating CNN dataset: {str(e)}", exc_info=True)
            issues.append(f"Validation error: {str(e)}")
            return {
                'is_valid': False,
                'issues': issues,
                'warnings': warnings,
                'info': info,
                'can_auto_fix': False,
                'fix_suggestions': fix_suggestions
            }
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'info': info,
            'can_auto_fix': can_auto_fix,
            'fix_suggestions': fix_suggestions
        }
    
    def _validate_rnn_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """Validate dataset for RNN (sequence data - text or tabular sequences)"""
        issues = []
        warnings = []
        info = {}
        fix_suggestions = []
        
        try:
            # RNN can work with CSV (tabular sequences) or text files
            ext = os.path.splitext(dataset_path)[1].lower()
            
            if ext not in ['.csv', '.txt', '.json']:
                issues.append(f"RNN requires CSV, TXT, or JSON format, found: {ext}")
                fix_suggestions.append("Convert file to CSV or TXT format")
                return {
                    'is_valid': False,
                    'issues': issues,
                    'warnings': warnings,
                    'info': info,
                    'can_auto_fix': False,
                    'fix_suggestions': fix_suggestions
                }
            
            if ext == '.csv':
                # Validate as tabular sequence data
                return self._validate_rnn_csv(dataset_path, issues, warnings, info, fix_suggestions)
            else:
                # Text file validation
                return self._validate_rnn_text(dataset_path, issues, warnings, info, fix_suggestions)
                
        except Exception as e:
            logger.error(f"Error validating RNN dataset: {str(e)}", exc_info=True)
            issues.append(f"Validation error: {str(e)}")
            return {
                'is_valid': False,
                'issues': issues,
                'warnings': warnings,
                'info': info,
                'can_auto_fix': False,
                'fix_suggestions': fix_suggestions
            }
    
    def _validate_rnn_csv(self, dataset_path: str, issues: List, warnings: List, info: Dict, fix_suggestions: List) -> Dict[str, Any]:
        """Validate RNN CSV dataset (similar to Decision Tree but sequence-aware)"""
        try:
            df = pd.read_csv(dataset_path, encoding='utf-8', nrows=1000)
            
            if df.empty:
                issues.append("Dataset is empty (no rows found)")
                return {
                    'is_valid': False,
                    'issues': issues,
                    'warnings': warnings,
                    'info': info,
                    'can_auto_fix': False,
                    'fix_suggestions': fix_suggestions
                }
            
            total_rows = sum(1 for _ in open(dataset_path, encoding='utf-8')) - 1
            if total_rows < 10:
                issues.append(f"Insufficient sequences: {total_rows} rows. Minimum required: 10")
            
            if len(df.columns) < 2:
                issues.append(f"Dataset must have at least 2 columns (features + target), found: {len(df.columns)}")
            
            # Check for sequence consistency
            target_col = df.columns[-1]
            missing_target = df[target_col].isnull().sum()
            
            if missing_target > 0:
                issues.append(f"Target column has {missing_target} missing values")
                fix_suggestions.append("Drop rows with missing target values")
            
            # Check if features can be sequences
            feature_cols = df.columns[:-1]
            if len(feature_cols) < 3:
                warnings.append(f"Only {len(feature_cols)} feature columns. RNN works better with sequence data (multiple time steps)")
            
            info.update({
                'num_sequences': total_rows,
                'sequence_length': len(feature_cols),
                'num_features': len(feature_cols),
                'target_column': target_col,
                'missing_values': int(missing_target)
            })
            
        except Exception as e:
            issues.append(f"Error reading CSV: {str(e)}")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'info': info,
            'can_auto_fix': bool(fix_suggestions),
            'fix_suggestions': fix_suggestions
        }
    
    def _validate_rnn_text(self, dataset_path: str, issues: List, warnings: List, info: Dict, fix_suggestions: List) -> Dict[str, Any]:
        """Validate RNN text dataset"""
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()[:1000]  # Sample
            
            if not lines:
                issues.append("Text file is empty")
                return {
                    'is_valid': False,
                    'issues': issues,
                    'warnings': warnings,
                    'info': info,
                    'can_auto_fix': False,
                    'fix_suggestions': fix_suggestions
                }
            
            # Check for valid text characters
            invalid_chars = []
            for i, line in enumerate(lines[:100]):
                try:
                    line.encode('utf-8')
                except:
                    invalid_chars.append(f"Line {i+1} contains invalid characters")
            
            if invalid_chars:
                warnings.append(f"Found {len(invalid_chars)} lines with encoding issues")
                fix_suggestions.append("Clean text and ensure UTF-8 encoding")
            
            info.update({
                'num_lines': len(lines),
                'avg_line_length': sum(len(line) for line in lines) / len(lines) if lines else 0,
                'has_invalid_chars': len(invalid_chars) > 0
            })
            
        except Exception as e:
            issues.append(f"Error reading text file: {str(e)}")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'info': info,
            'can_auto_fix': bool(fix_suggestions),
            'fix_suggestions': fix_suggestions
        }
    
    def generate_validation_report(self, validation_result: Dict[str, Any]) -> str:
        """Generate human-friendly validation report"""
        report_lines = [
            f"📊 DATASET VALIDATION REPORT",
            f"{'=' * 60}",
            f"Model Selected: {validation_result['model_type'].upper()}",
            f"Dataset Path: {validation_result['dataset_path']}",
            f"Status: {validation_result['status']}",
            f"",
        ]
        
        if validation_result['info']:
            report_lines.append("📋 Dataset Information:")
            for key, value in validation_result['info'].items():
                report_lines.append(f"  • {key.replace('_', ' ').title()}: {value}")
            report_lines.append("")
        
        if validation_result['issues']:
            report_lines.append("❌ Issues Found:")
            for i, issue in enumerate(validation_result['issues'], 1):
                report_lines.append(f"  {i}. {issue}")
            report_lines.append("")
        
        if validation_result['warnings']:
            report_lines.append("⚠️  Warnings:")
            for i, warning in enumerate(validation_result['warnings'], 1):
                report_lines.append(f"  {i}. {warning}")
            report_lines.append("")
        
        if validation_result['fix_suggestions']:
            report_lines.append("🔧 Suggested Fixes:")
            for i, fix in enumerate(validation_result['fix_suggestions'], 1):
                report_lines.append(f"  {i}. {fix}")
            report_lines.append("")
        
        if validation_result['can_auto_fix']:
            report_lines.append("✅ These issues can be automatically fixed")
        else:
            report_lines.append("⚠️  Manual fixes required")
        
        return "\n".join(report_lines)

