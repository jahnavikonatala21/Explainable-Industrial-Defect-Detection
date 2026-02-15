"""
Visualization Module for Defect Detection System

Creates visualizations for training, evaluation, and predictions
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config.config import *


class DefectVisualizer:
    """
    Handles all visualization operations
    """
    
    def __init__(self, class_names=CLASS_NAMES, output_dir=OUTPUTS_DIR):
        """
        Initialize visualizer
        
        Args:
            class_names: List of class names
            output_dir: Directory to save visualizations
        """
        self.class_names = class_names
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = DPI
        
    def plot_training_history(self, history, save_path=None):
        """
        Plot training history (loss and accuracy)
        
        Args:
            history: Keras training history object
            save_path: Path to save plot (None = auto-generate)
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'training_history.png')
        
        fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_LARGE)
        
        # Plot loss
        axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
        axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)
        
        # Plot accuracy
        axes[1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].legend(loc='best')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close()
        
        print(f"Training history plot saved to {save_path}")
        
    def plot_confusion_matrix(self, cm, save_path=None, normalize=False):
        """
        Plot confusion matrix heatmap
        
        Args:
            cm: Confusion matrix array
            save_path: Path to save plot
            normalize: Whether to normalize values
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'confusion_matrix.png')
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        plt.figure(figsize=FIGSIZE_MEDIUM)
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
        
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to {save_path}")
        
    def plot_sample_predictions(self, images, true_labels, predictions, 
                               num_samples=16, save_path=None):
        """
        Plot grid of sample predictions
        
        Args:
            images: Array of images
            true_labels: True class indices
            predictions: Prediction probabilities
            num_samples: Number of samples to display
            save_path: Path to save plot
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'sample_predictions.png')
        
        num_samples = min(num_samples, len(images))
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Calculate grid size
        grid_size = int(np.ceil(np.sqrt(num_samples)))
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        axes = axes.flatten()
        
        for i in range(num_samples):
            img = images[i]
            true_class = true_labels[i]
            pred_class = predicted_classes[i]
            confidence = predictions[i][pred_class]
            
            # Display image
            axes[i].imshow(img)
            axes[i].axis('off')
            
            # Color: green if correct, red if incorrect
            color = 'green' if true_class == pred_class else 'red'
            
            # Title with prediction info
            title = f"True: {self.class_names[true_class]}\n"
            title += f"Pred: {self.class_names[pred_class]}\n"
            title += f"Conf: {confidence:.2f}"
            
            axes[i].set_title(title, fontsize=9, color=color, fontweight='bold')
        
        # Hide remaining subplots
        for i in range(num_samples, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Sample Predictions', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close()
        
        print(f"Sample predictions saved to {save_path}")
        
    def plot_single_prediction(self, image, prediction, true_label=None, save_path=None):
        """
        Plot single image with prediction details
        
        Args:
            image: Single image array
            prediction: Prediction probabilities
            true_label: True class index (optional)
            save_path: Path to save plot
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'single_prediction.png')
        
        predicted_class = np.argmax(prediction)
        confidence = prediction[predicted_class]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_MEDIUM)
        
        # Display image
        ax1.imshow(image)
        ax1.axis('off')
        
        title = f"Predicted: {self.class_names[predicted_class]}\n"
        title += f"Confidence: {confidence:.2%}"
        
        if true_label is not None:
            color = 'green' if true_label == predicted_class else 'red'
            title = f"True: {self.class_names[true_label]}\n" + title
        else:
            color = 'blue'
        
        ax1.set_title(title, fontsize=12, color=color, fontweight='bold')
        
        # Plot prediction probabilities
        colors = ['green' if i == predicted_class else 'skyblue' 
                 for i in range(len(self.class_names))]
        
        ax2.barh(self.class_names, prediction, color=colors)
        ax2.set_xlabel('Probability', fontsize=12)
        ax2.set_title('Prediction Probabilities', fontsize=12, fontweight='bold')
        ax2.set_xlim([0, 1])
        
        # Add value labels on bars
        for i, v in enumerate(prediction):
            ax2.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close()
        
        print(f"Prediction visualization saved to {save_path}")
        
    def plot_class_distribution(self, labels, save_path=None):
        """
        Plot class distribution
        
        Args:
            labels: Array of class labels
            save_path: Path to save plot
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'class_distribution.png')
        
        unique, counts = np.unique(labels, return_counts=True)
        
        plt.figure(figsize=FIGSIZE_MEDIUM)
        bars = plt.bar([self.class_names[i] for i in unique], counts, 
                      color='steelblue', edgecolor='black', linewidth=1.5)
        
        plt.title('Class Distribution', fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Number of Samples', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close()
        
        print(f"Class distribution plot saved to {save_path}")
        
    def plot_roc_curves(self, roc_metrics, save_path=None):
        """
        Plot ROC curves for all classes
        
        Args:
            roc_metrics: Dictionary with ROC data per class
            save_path: Path to save plot
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'roc_curves.png')
        
        plt.figure(figsize=FIGSIZE_MEDIUM)
        
        for class_name, metrics in roc_metrics.items():
            plt.plot(metrics['fpr'], metrics['tpr'], 
                    label=f"{class_name} (AUC = {metrics['auc']:.3f})",
                    linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
        
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Multi-Class', fontsize=14, fontweight='bold', pad=20)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close()
        
        print(f"ROC curves saved to {save_path}")
    
    def plot_heatmap(self, image, heatmap_overlay, save_path=None):
        """
        Plot heatmap overlay on image
        
        Args:
            image: Original image
            heatmap_overlay: Heatmap overlay from GradCAM
            save_path: Path to save plot
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'heatmap.png')
        
        plt.figure(figsize=FIGSIZE_MEDIUM)
        plt.imshow(heatmap_overlay)
        plt.axis('off')
        plt.title('Defect Heatmap', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close()
        
        print(f"Heatmap visualization saved to {save_path}")
    
    def plot_prediction_with_heatmap(self, image, heatmap_overlay, prediction, 
                                    true_label=None, save_path=None):
        """
        Plot side-by-side comparison: original image, heatmap, and prediction probabilities
        
        Args:
            image: Original image
            heatmap_overlay: Heatmap overlay from GradCAM
            prediction: Prediction probabilities
            true_label: True class index (optional)
            save_path: Path to save plot
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'prediction_with_heatmap.png')
        
        predicted_class = np.argmax(prediction)
        confidence = prediction[predicted_class]
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot original image
        ax1.imshow(image)
        ax1.axis('off')
        ax1.set_title('Original Image', fontsize=12, fontweight='bold')
        
        # Plot heatmap overlay
        ax2.imshow(heatmap_overlay)
        ax2.axis('off')
        ax2.set_title('Defect Heatmap', fontsize=12, fontweight='bold')
        
        # Plot prediction probabilities
        colors_list = ['green' if i == predicted_class else 'skyblue' 
                      for i in range(len(self.class_names))]
        
        ax3.barh(self.class_names, prediction, color=colors_list)
        ax3.set_xlabel('Probability', fontsize=11)
        ax3.set_title('Predictions', fontsize=12, fontweight='bold')
        ax3.set_xlim([0, 1])
        
        # Add value labels
        for i, v in enumerate(prediction):
            ax3.text(v + 0.02, i, f'{v:.2f}', va='center', fontsize=9)
        
        # Overall title
        title = f"Predicted: {self.class_names[predicted_class]} ({confidence:.1%})"
        if true_label is not None:
            color = 'green' if true_label == predicted_class else 'red'
            title = f"True: {self.class_names[true_label]} | " + title
            fig.suptitle(title, fontsize=14, fontweight='bold', color=color)
        else:
            fig.suptitle(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close()
        
        print(f"Prediction with heatmap saved to {save_path}")


def visualize_training_results(history, confusion_matrix, output_dir=OUTPUTS_DIR):
    """
    Convenient function to visualize training results
    
    Args:
        history: Training history
        confusion_matrix: Confusion matrix array
        output_dir: Output directory
    """
    visualizer = DefectVisualizer(output_dir=output_dir)
    
    visualizer.plot_training_history(history)
    visualizer.plot_confusion_matrix(confusion_matrix)
    
    print(f"\nAll visualizations saved to {output_dir}")
