"""
Configuration settings for the Defect Detection System
"""
import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Dataset paths
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'validation')
TEST_DIR = os.path.join(DATA_DIR, 'test')
SAMPLES_DIR = os.path.join(DATA_DIR, 'samples')

# Defect classes
CLASS_NAMES = ['no_defect', 'defect', 'contamination', 'scratch', 'bend', 'cut']
NUM_CLASSES = 22  # Updated automatically

# Model configuration
MODEL_ARCHITECTURES = ['MobileNetV2', 'ResNet50']
DEFAULT_MODEL = 'MobileNetV2'  # Recommended for production
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# Training hyperparameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1

# Data augmentation parameters
AUGMENTATION_CONFIG = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'shear_range': 0.2,
    'zoom_range': 0.2,
    'horizontal_flip': True,
    'vertical_flip': False,
    'fill_mode': 'nearest'
}

# Training callbacks
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-7

# Model checkpoint
CHECKPOINT_PATH = os.path.join(MODELS_DIR, 'best_model.h5')
FINAL_MODEL_PATH = os.path.join(MODELS_DIR, 'final_model.h5')

# Visualization settings
FIGSIZE_LARGE = (12, 8)
FIGSIZE_MEDIUM = (10, 6)
FIGSIZE_SMALL = (8, 6)
DPI = 100

# Prediction settings
CONFIDENCE_THRESHOLD = 0.7  # Increased from 0.5 to reduce false positives
TOP_K_PREDICTIONS = 3

# Performance metrics
METRICS = ['accuracy', 'precision', 'recall']

# Random seed for reproducibility
RANDOM_SEED = 42
