"""
Data Preprocessing Module for Defect Detection System

This module handles image loading, preprocessing, augmentation, and dataset preparation.
"""
import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import *


class DataPreprocessor:
    """
    Handles all data preprocessing operations
    """
    
    def __init__(self, img_height=IMG_HEIGHT, img_width=IMG_WIDTH):
        """
        Initialize preprocessor
        
        Args:
            img_height: Target image height
            img_width: Target image width
        """
        self.img_height = img_height
        self.img_width = img_width
        self.input_shape = (img_height, img_width, IMG_CHANNELS)
        
    def load_and_preprocess_image(self, image_path, normalize=True):
        """
        Load and preprocess a single image
        
        Args:
            image_path: Path to image file
            normalize: Whether to normalize pixel values to [0, 1]
            
        Returns:
            Preprocessed image array
        """
        # Load image using OpenCV
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, (self.img_width, self.img_height))
        
        # Normalize if requested
        if normalize:
            img = img.astype(np.float32) / 255.0
            
        return img
    
    def load_dataset_from_directory(self, data_dir, class_names=CLASS_NAMES):
        """
        Load dataset from directory structure
        Expected structure: data_dir/class_name/image_files
        
        Args:
            data_dir: Root directory containing class subdirectories
            class_names: List of class names
            
        Returns:
            images: NumPy array of images
            labels: NumPy array of labels
            class_names: List of class names
        """
        images = []
        labels = []
        
        if not os.path.exists(data_dir):
            print(f"Warning: Directory {data_dir} does not exist")
            return np.array([]), np.array([]), class_names
        
        for class_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(data_dir, class_name)
            
            if not os.path.exists(class_dir):
                print(f"Warning: Class directory {class_dir} does not exist")
                continue
            
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            
            print(f"Loading {len(image_files)} images from {class_name}")
            
            for img_file in image_files:
                img_path = os.path.join(class_dir, img_file)
                try:
                    img = self.load_and_preprocess_image(img_path)
                    images.append(img)
                    labels.append(class_idx)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    continue
        
        if len(images) == 0:
            return np.array([]), np.array([]), class_names
        
        return np.array(images), np.array(labels), class_names
    
    def create_data_augmentation_generator(self):
        """
        Create ImageDataGenerator for data augmentation
        
        Returns:
            ImageDataGenerator configured for augmentation
        """
        augmentation_gen = ImageDataGenerator(
            rotation_range=AUGMENTATION_CONFIG['rotation_range'],
            width_shift_range=AUGMENTATION_CONFIG['width_shift_range'],
            height_shift_range=AUGMENTATION_CONFIG['height_shift_range'],
            shear_range=AUGMENTATION_CONFIG['shear_range'],
            zoom_range=AUGMENTATION_CONFIG['zoom_range'],
            horizontal_flip=AUGMENTATION_CONFIG['horizontal_flip'],
            vertical_flip=AUGMENTATION_CONFIG['vertical_flip'],
            fill_mode=AUGMENTATION_CONFIG['fill_mode'],
            preprocessing_function=None  # Already normalized
        )
        
        return augmentation_gen
    
    def create_validation_generator(self):
        """
        Create ImageDataGenerator for validation (no augmentation)
        
        Returns:
            ImageDataGenerator without augmentation
        """
        validation_gen = ImageDataGenerator()
        return validation_gen
    
    def split_dataset(self, X, y, val_split=VALIDATION_SPLIT, test_split=TEST_SPLIT, 
                     random_state=RANDOM_SEED):
        """
        Split dataset into train, validation, and test sets
        
        Args:
            X: Images array
            y: Labels array
            val_split: Validation split ratio
            test_split: Test split ratio
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_split, random_state=random_state, stratify=y
        )
        
        # Second split: separate validation from training
        val_size_adjusted = val_split / (1 - test_split)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=random_state, stratify=y_temp
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def prepare_labels(self, y, num_classes=NUM_CLASSES):
        """
        Convert labels to categorical (one-hot encoding)
        
        Args:
            y: Label array
            num_classes: Number of classes
            
        Returns:
            One-hot encoded labels
        """
        return to_categorical(y, num_classes=num_classes)
    
    def get_data_generators_from_directory(self, train_dir, val_dir, batch_size=BATCH_SIZE):
        """
        Create data generators from directory structure
        
        Args:
            train_dir: Training data directory
            val_dir: Validation data directory
            batch_size: Batch size
            
        Returns:
            train_generator, validation_generator
        """
        # Training data generator with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=AUGMENTATION_CONFIG['rotation_range'],
            width_shift_range=AUGMENTATION_CONFIG['width_shift_range'],
            height_shift_range=AUGMENTATION_CONFIG['height_shift_range'],
            shear_range=AUGMENTATION_CONFIG['shear_range'],
            zoom_range=AUGMENTATION_CONFIG['zoom_range'],
            horizontal_flip=AUGMENTATION_CONFIG['horizontal_flip'],
            vertical_flip=AUGMENTATION_CONFIG['vertical_flip'],
            fill_mode=AUGMENTATION_CONFIG['fill_mode']
        )
        
        # Validation data generator (only rescaling)
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        validation_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        return train_generator, validation_generator


def remove_background_grabcut(image):
    """
    Remove background using GrabCut algorithm to focus on main object
    
    Args:
        image: Input image (RGB, uint8 format)
        
    Returns:
        Image with background removed (set to white)
    """
    try:
        # Initialize mask
        mask = np.zeros(image.shape[:2], np.uint8)
        
        # Define rectangle around likely object area
        # Assume object is roughly centered, leaving 15% margin
        h, w = image.shape[:2]
        margin_h, margin_w = int(h * 0.15), int(w * 0.15)
        rect = (margin_w, margin_h, w - 2*margin_w, h - 2*margin_h)
        
        # Ensure rect is valid
        if rect[2] <= 0 or rect[3] <= 0:
            return image  # Return original if rect is invalid
        
        # GrabCut models
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # Apply GrabCut
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 
                    5, cv2.GC_INIT_WITH_RECT)
        
        # Create binary mask (0=background, 1=foreground)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        
        # Apply mask to image
        result = image.copy()
        result[mask2 == 0] = [255, 255, 255]  # Set background to white
        
        return result
        
    except Exception as e:
        print(f"Background removal failed: {e}")
        return image  # Return original image if GrabCut fails


def preprocess_single_image(image_path, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, remove_bg=False):
    """
    Standalone function to preprocess a single image for prediction
    
    Args:
        image_path: Path to image
        img_height: Target height
        img_width: Target width
        remove_bg: Whether to remove background using GrabCut (default: False)
        
    Returns:
        Preprocessed image ready for model input
    """
    # Load image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Remove background if requested
    if remove_bg:
        img = remove_background_grabcut(img)
    
    # Resize
    img = cv2.resize(img, (img_width, img_height))
    
    # Normalize
    img = img.astype(np.float32) / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img
