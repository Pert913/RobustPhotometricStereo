"""
Light Direction Predictor Package
=================================
ML and DL-based prediction of light directions for Photometric Stereo.

Available Models:
-----------------
Classical ML (sklearn):
  - ridge: Ridge Regression
  - rf: Random Forest
  - gbr: Gradient Boosting Regressor
  - mlp: Multi-Layer Perceptron

Deep Learning (PyTorch):
  - cnn: Custom Lightweight CNN
  - resnet: ResNet18 Transfer Learning
  - efficientnet: EfficientNet-B0 Transfer Learning

Usage:
------
    from light_direction_predictor import LightDirectionPredictor

    # ML model
    predictor = LightDirectionPredictor()
    X, Y, groups, _ = predictor.load_training_data('./data/training/')
    predictor.train(X, Y, model_type='rf')

    # DL model (requires PyTorch)
    predictor.train(X, Y, model_type='resnet', img_paths=img_paths, epochs=100)
"""

from .light_direction_predictor import (
    LightDirectionPredictor,
    compare_all_models,
    compare_ml_models,
    compare_dl_models,
    train_best_ml,
    train_best_dl,
    ML_MODELS,
    DL_MODELS,
    ALL_MODELS,
)

# Try to import DL components (optional)
try:
    from .dl_models import (
        DeepLightPredictor,
        LightCNN,
        ResNetLight,
        EfficientNetLight,
        TORCH_AVAILABLE,
    )
except ImportError:
    TORCH_AVAILABLE = False
    DeepLightPredictor = None
    LightCNN = None
    ResNetLight = None
    EfficientNetLight = None

__all__ = [
    'LightDirectionPredictor',
    'compare_all_models',
    'compare_ml_models',
    'compare_dl_models',
    'train_best_ml',
    'train_best_dl',
    'ML_MODELS',
    'DL_MODELS',
    'ALL_MODELS',
    'DeepLightPredictor',
    'LightCNN',
    'ResNetLight',
    'EfficientNetLight',
    'TORCH_AVAILABLE',
]
