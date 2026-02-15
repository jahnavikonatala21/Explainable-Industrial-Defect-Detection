"""
Evaluation Module for Defect Detection System

Comprehensive model evaluation with multiple metrics and analysis
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score,
    roc_curve
)
from config.config import *
from src.utils import print_section_header, format_metric


class ModelEvaluator:
    """
    Handles model evaluation and metrics calculation
    """
    
    def __init__(self, model, class_names=CLASS_NAMES):
        """
        Initialize evaluator
        
        Args:
            model: Trained Keras model
            class_names: List of class names
        """
        self.model = model
        self.class_names = class_names
        self.num_classes = len(class_names)
        
    def predict(self, X):
        """
        Make predictions on input data
        
        Args:
            X: Input images
            
        Returns:
            Predictions (probabilities) and predicted class indices
        """
        predictions = self.model.predict(X)
        predicted_classes = np.argmax(predictions, axis=1)
        
        return predictions, predicted_classes
    
    def evaluate(self, X_test, y_test):
        """
        Comprehensive model evaluation
        
        Args:
            X_test: Test images
            y_test: Test labels (one-hot encoded or class indices)
            
        Returns:
            Dictionary of evaluation metrics
        """
        print_section_header("Model Evaluation")
        
        # Convert one-hot to class indices if needed
        if len(y_test.shape) > 1 and y_test.shape[1] > 1:
            y_true = np.argmax(y_test, axis=1)
        else:
            y_true = y_test
        
        # Get predictions
        predictions, y_pred = self.predict(X_test)
        
        # Calculate metrics
        metrics = {}
        
        # Overall accuracy
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Precision, Recall, F1 (weighted averages)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        metrics['precision_per_class'] = precision_score(y_true, y_pred, average=None, zero_division=0)
        metrics['recall_per_class'] = recall_score(y_true, y_pred, average=None, zero_division=0)
        metrics['f1_per_class'] = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # Classification report
        metrics['classification_report'] = classification_report(
            y_true, y_pred, 
            target_names=self.class_names,
            zero_division=0
        )
        
        # Print results
        self.print_evaluation_results(metrics)
        
        return metrics
    
    def print_evaluation_results(self, metrics):
        """
        Print evaluation results in a formatted way
        
        Args:
            metrics: Dictionary of metrics
        """
        print("\n" + "="*60)
        print("EVALUATION RESULTS".center(60))
        print("="*60 + "\n")
        
        print(f"Overall Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Weighted Precision: {metrics['precision']:.4f}")
        print(f"Weighted Recall:    {metrics['recall']:.4f}")
        print(f"Weighted F1-Score:  {metrics['f1_score']:.4f}")
        
        print("\n" + "-"*60)
        print("Per-Class Metrics:")
        print("-"*60)
        
        for i, class_name in enumerate(self.class_names):
            print(f"\n{class_name}:")
            print(f"  Precision: {metrics['precision_per_class'][i]:.4f}")
            print(f"  Recall:    {metrics['recall_per_class'][i]:.4f}")
            print(f"  F1-Score:  {metrics['f1_per_class'][i]:.4f}")
        
        print("\n" + "="*60)
        print("Classification Report:")
        print("="*60)
        print(metrics['classification_report'])
        
    def get_confusion_matrix(self, X_test, y_test):
        """
        Get confusion matrix
        
        Args:
            X_test: Test images
            y_test: Test labels
            
        Returns:
            Confusion matrix array
        """
        # Convert one-hot to class indices if needed
        if len(y_test.shape) > 1 and y_test.shape[1] > 1:
            y_true = np.argmax(y_test, axis=1)
        else:
            y_true = y_test
        
        _, y_pred = self.predict(X_test)
        cm = confusion_matrix(y_true, y_pred)
        
        return cm
    
    def calculate_roc_metrics(self, X_test, y_test):
        """
        Calculate ROC curve and AUC for multi-class classification
        
        Args:
            X_test: Test images
            y_test: Test labels (one-hot encoded)
            
        Returns:
            Dictionary with ROC curves and AUC scores per class
        """
        # Get predictions
        predictions, _ = self.predict(X_test)
        
        # Ensure y_test is one-hot encoded
        if len(y_test.shape) == 1 or y_test.shape[1] == 1:
            from tensorflow.keras.utils import to_categorical
            y_test = to_categorical(y_test, num_classes=self.num_classes)
        
        roc_metrics = {}
        
        # Calculate ROC and AUC for each class
        for i in range(self.num_classes):
            fpr, tpr, thresholds = roc_curve(y_test[:, i], predictions[:, i])
            auc = roc_auc_score(y_test[:, i], predictions[:, i])
            
            roc_metrics[self.class_names[i]] = {
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds,
                'auc': auc
            }
        
        return roc_metrics


def evaluate_model(model, X_test, y_test, class_names=CLASS_NAMES):
    """
    Convenient function to evaluate a model
    
    Args:
        model: Trained Keras model
        X_test: Test images
        y_test: Test labels
        class_names: List of class names
        
    Returns:
        Dictionary of evaluation metrics
    """
    evaluator = ModelEvaluator(model, class_names)
    metrics = evaluator.evaluate(X_test, y_test)
    
    return metrics
