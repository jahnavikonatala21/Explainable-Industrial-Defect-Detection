"""
Severity Scoring Module for Defect Detection System

Calculates severity scores for detected defects based on multiple factors
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from config.config import CLASS_NAMES


# Severity level thresholds (0-10 scale)
# Classification based on score:
#   0-3: GOOD
#   3-6: SLIGHTLY_DEFECTIVE
#   Above 6: DEFECTIVE
SEVERITY_LEVELS = {
    'DEFECTIVE': 6,       # > 6 is defective
    'SLIGHTLY_DEFECTIVE': 3,  # 3-6 is slightly defective
    'GOOD': 0             # 0-3 is good
}



def get_defect_base_weights():
    """
    Get base severity weights for each defect type
    This will be dynamically learned from dataset statistics
    
    Returns:
        Dictionary mapping defect class to base weight (0-1)
    """
    # Default weights (can be overridden by dataset analysis)
    # These represent inherent severity of defect types
    default_weights = {
        'no_defect': 0.0,
        'crack': 1.0,        # Most critical - structural integrity
        'deformity': 0.8,    # High - affects form and function
        'discoloration': 0.6, # Medium - mainly cosmetic/quality
        'scratch': 0.4       # Low - surface level
    }
    
    return default_weights


def calculate_severity(defect_class, confidence, heatmap_stats=None, 
                       custom_weights=None):
    """
    Calculate severity score for a detected defect
    
    The severity score is based on:
    1. Defect type (intrinsic severity)
    2. Prediction confidence (how certain the model is)
    3. Heatmap concentration (how localized vs. distributed the defect is)
    
    Args:
        defect_class: Predicted defect class name
        confidence: Prediction confidence (0-1)
        heatmap_stats: Dictionary with heatmap statistics (optional)
        custom_weights: Custom defect type weights (optional)
        
    Returns:
        Dictionary with severity_score (0-10) and severity_level
    """
    # No defect = no severity
    if defect_class == 'no_defect':
        return {
            'severity_score': 0.0,
            'severity_level': 'NONE'
        }
    
    # Get defect weights
    weights = custom_weights if custom_weights else get_defect_base_weights()
    
    # Get base weight for this defect type (0-1)
    base_weight = weights.get(defect_class, 0.5)
    
    # Component 1: Base severity from defect type (40% of score, 0-4 points)
    type_component = base_weight * 4.0
    
    # Component 2: Confidence component (40% of score, 0-4 points)
    # Higher confidence = higher severity
    confidence_component = confidence * 4.0
    
    # Component 3: Heatmap concentration (20% of score, 0-2 points)
    concentration_component = 0
    if heatmap_stats:
        # High concentration = more localized defect = potentially more severe
        # Low concentration = distributed = potentially less severe or false positive
        concentration = heatmap_stats.get('concentration', 0.5)
        max_activation = heatmap_stats.get('max_activation', 0.5)
        
        # Combine concentration and max activation
        concentration_score = (concentration * 0.6 + max_activation * 0.4)
        concentration_component = concentration_score * 2.0
    else:
        # No heatmap data, use moderate value
        concentration_component = 1.0
    
    # Total severity score (0-10)
    severity_score = type_component + confidence_component + concentration_component
    
    # Clamp to [0, 10]
    severity_score = max(0, min(10, severity_score))
    
    # Determine severity level
    severity_level = get_severity_level(severity_score)
    
    return {
        'severity_score': round(severity_score, 1),
        'severity_level': severity_level,
        'breakdown': {
            'type_component': round(type_component, 1),
            'confidence_component': round(confidence_component, 1),
            'concentration_component': round(concentration_component, 1)
        }
    }


def get_severity_level(score):
    """
    Convert severity score to severity level
    
    Args:
        score: Severity score (0-10)
        
    Returns:
        Severity level string:
            - GOOD: 0-3
            - SLIGHTLY_DEFECTIVE: 3-6
            - DEFECTIVE: Above 6
    """
    if score > SEVERITY_LEVELS['DEFECTIVE']:  # > 6
        return 'DEFECTIVE'
    elif score >= SEVERITY_LEVELS['SLIGHTLY_DEFECTIVE']:  # >= 3 and <= 6
        return 'SLIGHTLY_DEFECTIVE'
    else:  # 0-3
        return 'GOOD'


def get_severity_color(severity_level):
    """
    Get color code for severity level (for UI display)
    
    Args:
        severity_level: Severity level string
        
    Returns:
        Hex color code
    """
    colors = {
        'DEFECTIVE': '#dc3545',          # Red
        'SLIGHTLY_DEFECTIVE': '#ffc107', # Yellow/Orange
        'GOOD': '#28a745',               # Green
        'NONE': '#6c757d'                # Gray
    }
    
    return colors.get(severity_level, '#6c757d')


def analyze_dataset_severity_distribution(dataset_results):
    """
    Analyze severity distribution across a dataset
    This can be used to calibrate severity weights
    
    Args:
        dataset_results: List of prediction results with severity scores
        
    Returns:
        Dictionary with distribution statistics
    """
    if not dataset_results:
        return {}
    
    severity_scores = [r.get('severity_score', 0) for r in dataset_results]
    severity_levels = [r.get('severity_level', 'NONE') for r in dataset_results]
    
    # Count severity levels
    level_counts = {}
    for level in severity_levels:
        level_counts[level] = level_counts.get(level, 0) + 1
    
    # Calculate statistics
    stats = {
        'mean_score': np.mean(severity_scores),
        'median_score': np.median(severity_scores),
        'std_score': np.std(severity_scores),
        'min_score': np.min(severity_scores),
        'max_score': np.max(severity_scores),
        'level_distribution': level_counts,
        'total_samples': len(dataset_results)
    }
    
    return stats


def learn_weights_from_dataset(train_data_dir):
    """
    Learn optimal severity weights from dataset structure
    
    This analyzes the dataset to understand the relative frequency
    of each defect type and can adjust weights accordingly.
    
    Args:
        train_data_dir: Path to training data directory
        
    Returns:
        Dictionary of learned weights
    """
    import os
    
    # Count samples per class
    class_counts = {}
    
    if not os.path.exists(train_data_dir):
        return get_defect_base_weights()
    
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(train_data_dir, class_name)
        if os.path.exists(class_dir):
            num_samples = len([f for f in os.listdir(class_dir) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            class_counts[class_name] = num_samples
    
    if not class_counts:
        return get_defect_base_weights()
    
    # Use inverse frequency as indicator of severity
    # Rare defects might be more severe
    total_samples = sum(class_counts.values())
    
    learned_weights = {}
    for class_name in CLASS_NAMES:
        if class_name == 'no_defect':
            learned_weights[class_name] = 0.0
        else:
            count = class_counts.get(class_name, 1)
            # Inverse frequency, normalized
            frequency = count / total_samples if total_samples > 0 else 0.2
            # Rarer defects get higher base weight
            learned_weights[class_name] = 1.0 - frequency
    
    # Normalize to [0, 1] range (excluding no_defect)
    defect_weights = [v for k, v in learned_weights.items() if k != 'no_defect']
    if defect_weights:
        max_weight = max(defect_weights)
        if max_weight > 0:
            for class_name in learned_weights:
                if class_name != 'no_defect':
                    learned_weights[class_name] /= max_weight
    
    return learned_weights
