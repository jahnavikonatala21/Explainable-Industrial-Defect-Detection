"""
CNN Model Architecture Module for Defect Detection

Implements transfer learning using pre-trained models (ResNet50, MobileNetV2)
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.optimizers import Adam
from config.config import *


class DefectDetectionModel:
    """
    Defect detection model using transfer learning
    """
    
    def __init__(self, architecture='MobileNetV2', num_classes=NUM_CLASSES, 
                 input_shape=INPUT_SHAPE, learning_rate=LEARNING_RATE):
        """
        Initialize model
        
        Args:
            architecture: Base architecture ('MobileNetV2' or 'ResNet50')
            num_classes: Number of output classes
            input_shape: Input image shape
            learning_rate: Learning rate for optimizer
        """
        self.architecture = architecture
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.model = None
        
    def build_model(self, trainable_base=False):
        """
        Build model with transfer learning
        
        Args:
            trainable_base: Whether to make base model layers trainable
            
        Returns:
            Compiled Keras model
        """
        # Load pre-trained base model
        if self.architecture == 'ResNet50':
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif self.architecture == 'MobileNetV2':
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        else:
            raise ValueError(f"Unsupported architecture: {self.architecture}")
        
        # Freeze base model layers
        base_model.trainable = trainable_base
        
        # Build custom classification head
        inputs = keras.Input(shape=self.input_shape)
        
        # Base model
        x = base_model(inputs, training=False)
        
        # Global average pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dense layers
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax', name='predictions')(x)
        
        # Create model
        self.model = keras.Model(inputs=inputs, outputs=outputs, name=f'DefectDetection_{self.architecture}')
        
        return self.model
    
    def compile_model(self, optimizer=None, loss='categorical_crossentropy', metrics=None):
        """
        Compile the model
        
        Args:
            optimizer: Optimizer instance (default: Adam)
            loss: Loss function
            metrics: List of metrics
        """
        if optimizer is None:
            optimizer = Adam(learning_rate=self.learning_rate)
        
        if metrics is None:
            metrics = ['accuracy', 
                      keras.metrics.Precision(name='precision'),
                      keras.metrics.Recall(name='recall')]
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        print(f"Model compiled with {loss} loss and {len(metrics)} metrics")
        
    def get_model(self):
        """
        Get the built model
        
        Returns:
            Keras model
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        return self.model
    
    def print_summary(self):
        """
        Print model summary
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        self.model.summary()
    
    def unfreeze_base_layers(self, num_layers=None):
        """
        Unfreeze base model layers for fine-tuning
        
        Args:
            num_layers: Number of layers to unfreeze from the end (None = all)
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        
        # Get base model (first layer is input, second is base model)
        base_model = self.model.layers[1]
        
        if num_layers is None:
            # Unfreeze all layers
            base_model.trainable = True
            print(f"Unfroze all {len(base_model.layers)} base model layers")
        else:
            # Unfreeze last num_layers
            base_model.trainable = True
            for layer in base_model.layers[:-num_layers]:
                layer.trainable = False
            print(f"Unfroze last {num_layers} base model layers")
        
        # Recompile with lower learning rate for fine-tuning
        self.compile_model(
            optimizer=Adam(learning_rate=self.learning_rate / 10)
        )


def create_defect_detection_model(architecture='MobileNetV2', num_classes=NUM_CLASSES,
                                  input_shape=INPUT_SHAPE, learning_rate=LEARNING_RATE,
                                  trainable_base=False):
    """
    Factory function to create and compile a defect detection model
    
    Args:
        architecture: 'MobileNetV2' or 'ResNet50'
        num_classes: Number of defect classes
        input_shape: Input image shape
        learning_rate: Learning rate
        trainable_base: Whether to train base model layers
        
    Returns:
        Compiled Keras model
    """
    model_builder = DefectDetectionModel(
        architecture=architecture,
        num_classes=num_classes,
        input_shape=input_shape,
        learning_rate=learning_rate
    )
    
    model = model_builder.build_model(trainable_base=trainable_base)
    model_builder.compile_model()
    
    print(f"\n{'='*60}")
    print(f"Created {architecture} model with {num_classes} classes")
    print(f"Total parameters: {model.count_params():,}")
    print(f"{'='*60}\n")
    
    return model


def load_trained_model(model_path):
    """
    Load a trained model from file
    
    Args:
        model_path: Path to saved model
        
    Returns:
        Loaded Keras model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = keras.models.load_model(model_path)
    print(f"Model loaded from {model_path}")
    
    return model
