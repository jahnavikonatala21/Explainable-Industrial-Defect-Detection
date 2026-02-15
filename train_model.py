"""
Main Training Script for Defect Detection System

Complete end-to-end training pipeline
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from config.config import *
from src.data_preprocessing import DataPreprocessor
from src.model import create_defect_detection_model
from src.training import ModelTrainer
from src.evaluation import ModelEvaluator
from src.visualization import DefectVisualizer
from src.dataset_manager import DatasetManager
from src.utils import print_section_header, setup_logger


def main():
    """
    Main training pipeline
    """
    print_section_header("DEFECT DETECTION SYSTEM - TRAINING")
    
    # Setup logger
    logger = setup_logger('TrainingPipeline', os.path.join(LOGS_DIR, 'training_pipeline.log'))
    
    # Check if sample data exists
    samples_dir = os.path.join(DATA_DIR, 'samples')
    if not os.path.exists(samples_dir) or len(os.listdir(samples_dir)) == 0:
        print("No sample data found. Generating synthetic samples...")
        from src.dataset_manager import setup_sample_dataset
        setup_sample_dataset(num_samples_per_class=100)
    
    # 1. Load and preprocess data
    print_section_header("Step 1: Loading and Preprocessing Data")
    
    preprocessor = DataPreprocessor()
    
    # Load data
    X, y, class_names = preprocessor.load_dataset_from_directory(samples_dir)
    
    if len(X) == 0:
        logger.error("No data loaded. Please check your dataset.")
        print("Error: No data found. Please add images to the data directory.")
        return
    
    logger.info(f"Loaded {len(X)} images")
    print(f"Loaded {len(X)} images")
    
    # Split dataset
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_dataset(X, y)
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
    
    # Prepare labels (one-hot encoding)
    y_train_cat = preprocessor.prepare_labels(y_train)
    y_val_cat = preprocessor.prepare_labels(y_val)
    y_test_cat = preprocessor.prepare_labels(y_test)
    
    # 2. Create model
    print_section_header("Step 2: Creating Model")
    
    model = create_defect_detection_model(
        architecture=DEFAULT_MODEL,
        num_classes=NUM_CLASSES,
        learning_rate=LEARNING_RATE
    )
    
    logger.info(f"Created {DEFAULT_MODEL} model")
    
    # 3. Train model
    print_section_header("Step 3: Training Model")
    
    trainer = ModelTrainer(model)
    history = trainer.train(
        X_train, y_train_cat,
        X_val, y_val_cat,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )
    
    # Save final model
    trainer.save_final_model()
    logger.info("Training completed")
    
    # 4. Evaluate model
    print_section_header("Step 4: Evaluating Model")
    
    evaluator = ModelEvaluator(model, CLASS_NAMES)
    metrics = evaluator.evaluate(X_test, y_test_cat)
    
    # Get confusion matrix
    cm = evaluator.get_confusion_matrix(X_test, y_test_cat)
    
    # 5. Visualize results
    print_section_header("Step 5: Creating Visualizations")
    
    visualizer = DefectVisualizer()
    
    # Training history
    visualizer.plot_training_history(history)
    
    # Confusion matrix
    visualizer.plot_confusion_matrix(cm)
    visualizer.plot_confusion_matrix(cm, 
                                    save_path=os.path.join(OUTPUTS_DIR, 'confusion_matrix_normalized.png'),
                                    normalize=True)
    
    # Sample predictions
    test_predictions, _ = evaluator.predict(X_test)
    visualizer.plot_sample_predictions(X_test, y_test, test_predictions, num_samples=16)
    
    # Class distribution
    visualizer.plot_class_distribution(y_train)
    
    print_section_header("Training Pipeline Completed")
    
    print(f"\n{'='*60}")
    print("SUMMARY".center(60))
    print(f"{'='*60}")
    print(f"Model: {DEFAULT_MODEL}")
    print(f"Training Samples: {len(X_train)}")
    print(f"Validation Samples: {len(X_val)}")
    print(f"Test Samples: {len(X_test)}")
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test Precision: {metrics['precision']:.4f}")
    print(f"Test Recall: {metrics['recall']:.4f}")
    print(f"Test F1-Score: {metrics['f1_score']:.4f}")
    print(f"\nModel saved to: {FINAL_MODEL_PATH}")
    print(f"Best model saved to: {CHECKPOINT_PATH}")
    print(f"Visualizations saved to: {OUTPUTS_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
