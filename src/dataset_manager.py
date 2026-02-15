"""
Dataset Manager for Defect Detection System

Handles dataset organization, validation, and sample generation
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
from config.config import *
from src.utils import create_directory_structure


class DatasetManager:
    """
    Manages dataset operations
    """
    
    def __init__(self, data_dir=DATA_DIR):
        """
        Initialize dataset manager
        
        Args:
            data_dir: Root data directory
        """
        self.data_dir = data_dir
        
    def create_dataset_structure(self):
        """
        Create standard dataset directory structure
        """
        # Create main directories
        for split in ['train', 'validation', 'test', 'samples']:
            split_dir = os.path.join(self.data_dir, split)
            create_directory_structure(split_dir, CLASS_NAMES)
        
        print("Dataset directory structure created successfully")
        
    def validate_dataset(self, dataset_dir):
        """
        Validate dataset structure and contents
        
        Args:
            dataset_dir: Directory to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'class_counts': {}
        }
        
        # Check if directory exists
        if not os.path.exists(dataset_dir):
            validation_results['valid'] = False
            validation_results['errors'].append(f"Directory does not exist: {dataset_dir}")
            return validation_results
        
        # Check each class directory
        for class_name in CLASS_NAMES:
            class_dir = os.path.join(dataset_dir, class_name)
            
            if not os.path.exists(class_dir):
                validation_results['warnings'].append(f"Class directory missing: {class_name}")
                validation_results['class_counts'][class_name] = 0
                continue
            
            # Count images
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            
            validation_results['class_counts'][class_name] = len(image_files)
            
            if len(image_files) == 0:
                validation_results['warnings'].append(f"No images found in {class_name}")
        
        # Check class balance
        counts = list(validation_results['class_counts'].values())
        if len(counts) > 0:
            max_count = max(counts)
            min_count = min(counts)
            
            if max_count > 0 and min_count / max_count < 0.3:
                validation_results['warnings'].append(
                    f"Class imbalance detected. Max: {max_count}, Min: {min_count}"
                )
        
        return validation_results
    
    def print_validation_results(self, results):
        """
        Print validation results
        
        Args:
            results: Validation results dictionary
        """
        print("\n" + "="*60)
        print("Dataset Validation Results".center(60))
        print("="*60 + "\n")
        
        if results['valid']:
            print("✓ Dataset structure is valid")
        else:
            print("✗ Dataset validation failed")
        
        # Print class counts
        print("\nClass Distribution:")
        print("-"*60)
        for class_name, count in results['class_counts'].items():
            print(f"  {class_name:<20} {count:>5} images")
        
        total = sum(results['class_counts'].values())
        print("-"*60)
        print(f"  {'Total':<20} {total:>5} images\n")
        
        # Print errors
        if results['errors']:
            print("Errors:")
            for error in results['errors']:
                print(f"  ✗ {error}")
            print()
        
        # Print warnings
        if results['warnings']:
            print("Warnings:")
            for warning in results['warnings']:
                print(f"  ⚠ {warning}")
            print()
        
        print("="*60 + "\n")
    
    def generate_sample_images(self, num_samples_per_class=20):
        """
        Generate synthetic sample images for demonstration
        
        Args:
            num_samples_per_class: Number of samples to generate per class
        """
        samples_dir = os.path.join(self.data_dir, 'samples')
        
        print(f"Generating {num_samples_per_class} sample images per class...")
        
        for class_name in CLASS_NAMES:
            class_dir = os.path.join(samples_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            for i in range(num_samples_per_class):
                # Create base image
                img = self._generate_sample_image(class_name)
                
                # Save image
                filename = f"{class_name}_{i:03d}.jpg"
                filepath = os.path.join(class_dir, filename)
                cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
            print(f"  Generated {num_samples_per_class} samples for {class_name}")
        
        print("Sample generation completed!")
    
    def _generate_sample_image(self, defect_type):
        """
        Generate a synthetic sample image with simulated defect
        
        Args:
            defect_type: Type of defect to simulate
            
        Returns:
            Generated image array
        """
        # Create base image with random pattern
        img = np.random.randint(100, 200, (IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
        
        # Add gradient for more realistic appearance
        gradient = np.linspace(0, 50, IMG_WIDTH, dtype=np.uint8)
        img[:, :, 0] += gradient
        img[:, :, 1] += gradient[::-1]
        
        # Apply Gaussian blur for smoother appearance
        img = cv2.GaussianBlur(img, (5, 5), 0)
        
        # Add defect based on type
        if defect_type == 'scratch':
            img = self._add_scratch(img)
        elif defect_type == 'crack':
            img = self._add_crack(img)
        elif defect_type == 'discoloration':
            img = self._add_discoloration(img)
        elif defect_type == 'deformity':
            img = self._add_deformity(img)
        # 'no_defect' - keep as is
        
        return img
    
    def _add_scratch(self, img):
        """Add scratch defect to image"""
        h, w = img.shape[:2]
        # Random line representing scratch
        pt1 = (np.random.randint(0, w), np.random.randint(0, h))
        pt2 = (np.random.randint(0, w), np.random.randint(0, h))
        cv2.line(img, pt1, pt2, (50, 50, 50), 2)
        return img
    
    def _add_crack(self, img):
        """Add crack defect to image"""
        h, w = img.shape[:2]
        # Jagged line representing crack
        points = []
        for i in range(5):
            x = int(w * i / 4) + np.random.randint(-20, 20)
            y = int(h * i / 4) + np.random.randint(-20, 20)
            points.append((x, y))
        
        for i in range(len(points) - 1):
            cv2.line(img, points[i], points[i+1], (30, 30, 30), 3)
        
        return img
    
    def _add_discoloration(self, img):
        """Add discoloration defect to image"""
        h, w = img.shape[:2]
        # Random colored patch
        center = (np.random.randint(w//4, 3*w//4), np.random.randint(h//4, 3*h//4))
        radius = np.random.randint(20, 50)
        color = (np.random.randint(0, 100), np.random.randint(100, 255), 
                np.random.randint(0, 100))
        cv2.circle(img, center, radius, color, -1)
        # Blur to make it more natural
        img = cv2.GaussianBlur(img, (11, 11), 0)
        return img
    
    def _add_deformity(self, img):
        """Add structural deformity to image"""
        h, w = img.shape[:2]
        # Rectangular distortion
        x = np.random.randint(w//4, w//2)
        y = np.random.randint(h//4, h//2)
        w_rect = np.random.randint(30, 80)
        h_rect = np.random.randint(30, 80)
        
        # Create darker/lighter region
        intensity = np.random.choice([-50, 50])
        img[y:y+h_rect, x:x+w_rect] = np.clip(
            img[y:y+h_rect, x:x+w_rect].astype(int) + intensity, 0, 255
        ).astype(np.uint8)
        
        return img


def setup_sample_dataset(num_samples_per_class=20):
    """
    Setup sample dataset for demonstration
    
    Args:
        num_samples_per_class: Number of samples per class
    """
    manager = DatasetManager()
    manager.create_dataset_structure()
    manager.generate_sample_images(num_samples_per_class)
    
    print("\nSample dataset created successfully!")
