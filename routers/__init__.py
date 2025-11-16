"""
Routers module - API endpoint handlers
"""
from . import dataset, model, training, evaluation, compression, comparison

__all__ = ['dataset', 'model', 'training', 'evaluation', 'compression', 'comparison']