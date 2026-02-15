"""
Training Pipeline Module for Defect Detection System

Handles model training with callbacks, checkpointing, and monitoring
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tensorflow import keras
from tensorflow.keras.callbacks import (
    ModelCheckpoint, 
    EarlyStopping, 
    ReduceLROnPlateau, 
    CSVLogger,
    TensorBoard
)
from config.config import *
from src.utils import setup_logger, get_timestamp, print_section_header


class ModelTrainer:
    """
    Handles model training operations
    """
    
    def __init__(self, model, log_dir=LOGS_DIR):
        """
        Initialize trainer
        
        Args:
            model: Compiled Keras model
            log_dir: Directory for logs
        """
        self.model = model
        self.log_dir = log_dir
        self.logger = setup_logger('ModelTrainer', 
                                   os.path.join(log_dir, f'training_{get_timestamp()}.log'))
        self.history = None
        
    def get_callbacks(self, checkpoint_path=CHECKPOINT_PATH, 
                     early_stopping=True, reduce_lr=True,
                     tensorboard=False):
        """
        Create training callbacks
        
        Args:
            checkpoint_path: Path to save best model
            early_stopping: Whether to use early stopping
            reduce_lr: Whether to reduce learning rate on plateau
            tensorboard: Whether to use TensorBoard logging
            
        Returns:
            List of callbacks
        """
        callbacks = []
        
        # Model checkpoint - save best model
        checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callbacks.append(checkpoint_callback)
        self.logger.info(f"Model checkpoint: {checkpoint_path}")
        
        # Early stopping
        if early_stopping:
            early_stop_callback = EarlyStopping(
                monitor='val_loss',
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stop_callback)
            self.logger.info(f"Early stopping with patience: {EARLY_STOPPING_PATIENCE}")
        
        # Reduce learning rate on plateau
        if reduce_lr:
            reduce_lr_callback = ReduceLROnPlateau(
                monitor='val_loss',
                factor=REDUCE_LR_FACTOR,
                patience=REDUCE_LR_PATIENCE,
                min_lr=MIN_LR,
                verbose=1
            )
            callbacks.append(reduce_lr_callback)
            self.logger.info(f"Reduce LR on plateau - factor: {REDUCE_LR_FACTOR}, patience: {REDUCE_LR_PATIENCE}")
        
        # CSV logger
        csv_log_path = os.path.join(self.log_dir, f'training_log_{get_timestamp()}.csv')
        csv_logger = CSVLogger(csv_log_path, append=True)
        callbacks.append(csv_logger)
        self.logger.info(f"CSV logging: {csv_log_path}")
        
        # TensorBoard (optional)
        if tensorboard:
            tb_log_dir = os.path.join(self.log_dir, 'tensorboard', get_timestamp())
            tb_callback = TensorBoard(log_dir=tb_log_dir, histogram_freq=1)
            callbacks.append(tb_callback)
            self.logger.info(f"TensorBoard logging: {tb_log_dir}")
        
        return callbacks
    
    def train(self, X_train, y_train, X_val, y_val, 
             batch_size=BATCH_SIZE, epochs=EPOCHS,
             callbacks=None, class_weights=None, verbose=1):
        """
        Train the model
        
        Args:
            X_train: Training images
            y_train: Training labels
            X_val: Validation images
            y_val: Validation labels
            batch_size: Batch size
            epochs: Number of epochs
            callbacks: List of callbacks (None = default callbacks)
            class_weights: Dictionary of class weights
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        print_section_header("Starting Model Training")
        
        self.logger.info(f"Training samples: {len(X_train)}")
        self.logger.info(f"Validation samples: {len(X_val)}")
        self.logger.info(f"Batch size: {batch_size}")
        self.logger.info(f"Epochs: {epochs}")
        
        # Get default callbacks if none provided
        if callbacks is None:
            callbacks = self.get_callbacks()
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=verbose
        )
        
        self.logger.info("Training completed")
        
        return self.history
    
    def train_with_generator(self, train_generator, validation_generator,
                            epochs=EPOCHS, callbacks=None, 
                            steps_per_epoch=None, validation_steps=None,
                            class_weights=None, verbose=1):
        """
        Train model using data generators
        
        Args:
            train_generator: Training data generator
            validation_generator: Validation data generator
            epochs: Number of epochs
            callbacks: List of callbacks
            steps_per_epoch: Steps per epoch (None = auto-calculate)
            validation_steps: Validation steps (None = auto-calculate)
            class_weights: Dictionary of class weights
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        print_section_header("Starting Model Training (with generators)")
        
        # Get default callbacks if none provided
        if callbacks is None:
            callbacks = self.get_callbacks()
        
        # Train model
        self.history = self.model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=verbose
        )
        
        self.logger.info("Training completed")
        
        return self.history
    
    def save_final_model(self, save_path=FINAL_MODEL_PATH):
        """
        Save the final trained model
        
        Args:
            save_path: Path to save model
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.model.save(save_path)
        self.logger.info(f"Final model saved to {save_path}")
        print(f"Model saved to {save_path}")
    
    def get_training_history(self):
        """
        Get training history
        
        Returns:
            History object
        """
        return self.history


def train_defect_detection_model(model, X_train, y_train, X_val, y_val,
                                 epochs=EPOCHS, batch_size=BATCH_SIZE,
                                 checkpoint_path=CHECKPOINT_PATH):
    """
    Convenient function to train a defect detection model
    
    Args:
        model: Compiled Keras model
        X_train: Training images
        y_train: Training labels
        X_val: Validation images
        y_val: Validation labels
        epochs: Number of epochs
        batch_size: Batch size
        checkpoint_path: Path to save best model
        
    Returns:
        Training history and trainer instance
    """
    trainer = ModelTrainer(model)
    history = trainer.train(
        X_train, y_train, X_val, y_val,
        batch_size=batch_size,
        epochs=epochs
    )
    
    # Save final model
    trainer.save_final_model()
    
    return history, trainer
