"""
Utility functions for the Defect Detection System
"""
import os
import time
import json
import logging
from functools import wraps
from datetime import datetime
import numpy as np

def setup_logger(name, log_file=None, level=logging.INFO):
    """
    Set up logger with file and console handlers
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def timer(func):
    """
    Decorator to measure function execution time
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} executed in {elapsed_time:.2f} seconds")
        return result
    return wrapper


def save_json(data, filepath):
    """
    Save data to JSON file
    
    Args:
        data: Data to save
        filepath: Path to save file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Data saved to {filepath}")


def load_json(filepath):
    """
    Load data from JSON file
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Loaded data
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def create_directory_structure(base_dir, subdirs):
    """
    Create directory structure
    
    Args:
        base_dir: Base directory path
        subdirs: List of subdirectory names
    """
    for subdir in subdirs:
        path = os.path.join(base_dir, subdir)
        os.makedirs(path, exist_ok=True)
    print(f"Directory structure created at {base_dir}")


def get_timestamp():
    """
    Get current timestamp string
    
    Returns:
        Formatted timestamp string
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def print_section_header(title, char='='):
    """
    Print formatted section header
    
    Args:
        title: Section title
        char: Character for border
    """
    border = char * 60
    print(f"\n{border}")
    print(f"{title.center(60)}")
    print(f"{border}\n")


def calculate_class_weights(y_train):
    """
    Calculate class weights for imbalanced datasets
    
    Args:
        y_train: Training labels
        
    Returns:
        Dictionary of class weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))
    
    return class_weights


def format_metric(value, metric_name):
    """
    Format metric value for display
    
    Args:
        value: Metric value
        metric_name: Name of metric
        
    Returns:
        Formatted string
    """
    if isinstance(value, float):
        return f"{metric_name}: {value:.4f}"
    return f"{metric_name}: {value}"


def get_model_size(model_path):
    """
    Get model file size in MB
    
    Args:
        model_path: Path to model file
        
    Returns:
        Size in MB
    """
    size_bytes = os.path.getsize(model_path)
    size_mb = size_bytes / (1024 * 1024)
    return size_mb
