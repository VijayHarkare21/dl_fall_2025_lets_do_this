"""
Evaluation Module
=================

Contains evaluation code for assessing SSL representation quality.

- KNNEvaluator: k-Nearest Neighbors classification on frozen features
- create_predictions_csv: Generate Kaggle submission files
"""

from .knn import KNNEvaluator, create_predictions_csv

__all__ = [
    'KNNEvaluator',
    'create_predictions_csv',
]