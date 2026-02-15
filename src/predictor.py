"""
Prediction Module for Defect Detection System

Handles single and batch predictions for trained models
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
from config.config import *
from src.data_preprocessing import preprocess_single_image
from src.model import load_trained_model
from src.utils import print_section_header


class DefectPredictor:
    """
    Handles predictions for defect detection
    """
    
    def __init__(self, model_path, class_names=None):
        """
        Initialize predictor
        
        Args:
            model_path: Path to trained model
            class_names: List of class names (auto-loaded from models/class_names.txt if None)
        """
        self.model_path = model_path
        
        # Load class names from file if not provided
        if class_names is None:
            class_names_path = os.path.join(os.path.dirname(model_path), 'class_names.txt')
            if os.path.exists(class_names_path):
                with open(class_names_path, 'r') as f:
                    class_names = [line.strip() for line in f if line.strip()]
                print(f"Loaded {len(class_names)} class names from {class_names_path}")
            else:
                class_names = CLASS_NAMES
                print(f"Using default CLASS_NAMES ({len(class_names)} classes)")
        
        self.class_names = class_names
        self.model = load_trained_model(model_path)
        print(f"Model output classes: {self.model.output_shape[-1]}, Loaded class names: {len(self.class_names)}")
        
    def predict_single(self, image_path, top_k=TOP_K_PREDICTIONS, remove_bg=False,
                      include_heatmap=False, include_severity=False, data_dir=None):
        """
        Predict defect class for a single image
        
        Args:
            image_path: Path to image file
            top_k: Number of top predictions to return
            remove_bg: Whether to remove background before prediction (default: False)
            include_heatmap: Whether to generate GradCAM heatmap (default: False)
            include_severity: Whether to calculate severity score (default: False)
            data_dir: Path to training data for severity weight learning (optional)
            
        Returns:
            Dictionary with prediction results (includes heatmap_path and severity if requested)
        """
        # Preprocess image with optional background removal
        img = preprocess_single_image(image_path, remove_bg=remove_bg)
        
        # Make prediction
        predictions = self.model.predict(img, verbose=0)[0]
        
        # Detect actual number of classes from model output
        num_classes = len(predictions)
        
        # Adjust class_names if model has different number of classes
        if num_classes == 2:
            # Binary classification: no_defect vs defective
            actual_class_names = ['no_defect', 'defective']
        else:
            # Multi-class: use original class names
            actual_class_names = self.class_names[:num_classes]
        
        # Get top-k predictions (limited by actual number of classes)
        top_k = min(top_k, num_classes)
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        
        results = {
            'image_path': image_path,
            'predictions': [],
            'predicted_class': actual_class_names[top_indices[0]],
            'predicted_class_index': int(top_indices[0]),
            'confidence': float(predictions[top_indices[0]]),
            'all_probabilities': {actual_class_names[i]: float(predictions[i]) 
                                  for i in range(num_classes)}
        }
        
        # Add top-k predictions
        for idx in top_indices:
            results['predictions'].append({
                'class': actual_class_names[idx],
                'class_index': int(idx),
                'probability': float(predictions[idx])
            })
        
        # Generate heatmap if requested
        if include_heatmap:
            try:
                from src.gradcam import generate_gradcam
                import cv2
                
                # Determine if product is defective based on predicted class
                predicted_class = results['predicted_class']
                is_defective = predicted_class != 'no_defect'
                
                # Generate heatmap for predicted class
                # Use blue colormap for good products, red/hot for defects
                heatmap, overlay, heatmap_stats = generate_gradcam(
                    self.model, img, top_indices[0], is_defective=is_defective
                )
                
                # Save heatmap overlay
                heatmap_filename = f"heatmap_{os.path.basename(image_path)}"
                heatmap_path = os.path.join(OUTPUTS_DIR, heatmap_filename)
                
                # Convert overlay to proper format and save
                overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                cv2.imwrite(heatmap_path, overlay_bgr)
                
                results['heatmap_path'] = heatmap_path
                results['heatmap_stats'] = heatmap_stats
                
            except Exception as e:
                print(f"Warning: Could not generate heatmap: {e}")
                results['heatmap_path'] = None
                results['heatmap_stats'] = None
        
        # Calculate severity if requested
        if include_severity:
            try:
                from src.severity import calculate_severity, learn_weights_from_dataset
                
                # Learn weights from dataset if path provided
                custom_weights = None
                if data_dir:
                    custom_weights = learn_weights_from_dataset(data_dir)
                
                # Get heatmap stats if available from heatmap generation
                heatmap_stats = results.get('heatmap_stats', None)
                
                # Calculate severity
                severity_result = calculate_severity(
                    results['predicted_class'],
                    results['confidence'],
                    heatmap_stats,
                    custom_weights
                )
                
                results['severity_score'] = severity_result.get('severity_score', 0)
                results['severity_level'] = severity_result.get('severity_level', 'UNKNOWN')
                results['severity_breakdown'] = severity_result.get('breakdown', {})
                
            except Exception as e:
                print(f"Warning: Could not calculate severity: {e}")
                results['severity_score'] = 0
                results['severity_level'] = 'UNKNOWN'
                results['severity_breakdown'] = {}
        
        return results

    
    def predict_batch(self, image_paths):
        """
        Predict defect classes for multiple images
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of prediction results
        """
        results = []
        
        print(f"Processing {len(image_paths)} images...")
        
        for i, image_path in enumerate(image_paths):
            try:
                result = self.predict_single(image_path)
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(image_paths)} images")
                    
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        print(f"Completed processing {len(image_paths)} images")
        
        return results
    
    def predict_with_visualization(self, image_path, visualizer=None):
        """
        Predict and create visualization
        
        Args:
            image_path: Path to image
            visualizer: DefectVisualizer instance (optional)
            
        Returns:
            Prediction results
        """
        # Get prediction
        result = self.predict_single(image_path)
        
        # Create visualization if visualizer provided
        if visualizer:
            img = preprocess_single_image(image_path)[0]
            predictions = np.array([result['all_probabilities'][class_name] 
                                   for class_name in self.class_names])
            
            viz_path = os.path.join(OUTPUTS_DIR, 
                                   f"prediction_{os.path.basename(image_path)}")
            visualizer.plot_single_prediction(img, predictions, save_path=viz_path)
            
            result['visualization_path'] = viz_path
        
        return result
    
    def print_prediction_result(self, result):
        """
        Print prediction result in formatted way
        
        Args:
            result: Prediction result dictionary
        """
        print_section_header(f"Prediction for {os.path.basename(result['image_path'])}")
        
        if 'error' in result:
            print(f"Error: {result['error']}")
            return
        
        print(f"Predicted Class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"\nTop {len(result['predictions'])} Predictions:")
        print("-" * 50)
        
        for i, pred in enumerate(result['predictions'], 1):
            print(f"{i}. {pred['class']:<20} {pred['probability']:.2%}")
        
        print("=" * 60 + "\n")
    
    def is_defective(self, prediction_result, threshold=CONFIDENCE_THRESHOLD):
        """
        Determine if product is defective based on prediction.
        
        For 49-class MVTec model:
        - If predicted class is anything other than 'no_defect' → IS defective
        - If predicted class is 'no_defect' but confidence is low → Still defective
        
        Args:
            prediction_result: Prediction result dictionary
            threshold: Confidence threshold
            
        Returns:
            Boolean indicating if defective
        """
        predicted_class = prediction_result['predicted_class'].lower()
        confidence = prediction_result['confidence']
        all_probs = prediction_result['all_probabilities']
        
        # Get no_defect probability (check various possible names)
        no_defect_prob = 0
        for key in all_probs:
            if 'no_defect' in key.lower() or key.lower() == 'good':
                no_defect_prob = max(no_defect_prob, all_probs[key])
        
        # Debug output
        print(f"Classification Debug: predicted='{predicted_class}', confidence={confidence:.2%}, no_defect_prob={no_defect_prob:.2%}")
        
        # Rule 1: If predicted class contains 'no_defect' or 'good' with high confidence → NOT defective
        if ('no_defect' in predicted_class or predicted_class == 'good') and confidence >= 0.50:
            print(f"  → Classification: GOOD (high confidence no_defect)")
            return False
        
        # Rule 2: If no_defect probability is very high (>60%) → NOT defective
        if no_defect_prob >= 0.60:
            print(f"  → Classification: GOOD (no_defect prob >= 60%)")
            return False
        
        # Rule 3: If predicted class is any defect type → IS defective
        if 'no_defect' not in predicted_class and predicted_class != 'good':
            print(f"  → Classification: DEFECTIVE (predicted class is a defect type)")
            return True
        
        # Rule 4: If no_defect probability is low (<50%) → IS defective
        if no_defect_prob < 0.50:
            print(f"  → Classification: DEFECTIVE (no_defect prob < 50%)")
            return True
        
        # Default: NOT defective for ambiguous cases
        print(f"  → Classification: GOOD (default - ambiguous)")
        return False



def predict_image(model_path, image_path, verbose=True):
    """
    Convenient function to predict a single image
    
    Args:
        model_path: Path to trained model
        image_path: Path to image
        verbose: Whether to print results
        
    Returns:
        Prediction results
    """
    predictor = DefectPredictor(model_path)
    result = predictor.predict_single(image_path)
    
    if verbose:
        predictor.print_prediction_result(result)
    
    return result


def predict_images(model_path, image_paths, output_file=None):
    """
    Convenient function to predict multiple images
    
    Args:
        model_path: Path to trained model
        image_paths: List of image paths
        output_file: Optional path to save results (JSON)
        
    Returns:
        List of prediction results
    """
    predictor = DefectPredictor(model_path)
    results = predictor.predict_batch(image_paths)
    
    # Save results if output file specified
    if output_file:
        import json
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {output_file}")
    
    return results
