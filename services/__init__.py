"""
Services module - Business logic for ML operations
"""
from .training_service import TrainingService
from .evaluation_service import EvaluationService
from .compression_service import CompressionService

__all__ = ['TrainingService', 'EvaluationService', 'CompressionService']