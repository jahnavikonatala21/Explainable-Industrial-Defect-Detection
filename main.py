"""
Main Application Script for Defect Detection System

Unified command-line interface for all operations
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
from config.config import *


def parse_arguments():
    """
    Parse command line arguments
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Defect Detection System - Unified Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup sample dataset')
    setup_parser.add_argument('--samples', type=int, default=100,
                             help='Number of samples per class (default: 100)')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument('--model', type=str, choices=['MobileNetV2', 'ResNet50'],
                             default='MobileNetV2', help='Model architecture')
    train_parser.add_argument('--epochs', type=int, default=EPOCHS,
                             help=f'Number of epochs (default: {EPOCHS})')
    train_parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                             help=f'Batch size (default: {BATCH_SIZE})')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument('--image', type=str, help='Path to image')
    predict_parser.add_argument('--model', type=str, default=CHECKPOINT_PATH,
                               help='Path to model')
    predict_parser.add_argument('--visualize', action='store_true',
                               help='Create visualization')
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate model')
    evaluate_parser.add_argument('--model', type=str, default=CHECKPOINT_PATH,
                                help='Path to model')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate dataset')
    validate_parser.add_argument('--data-dir', type=str, default=SAMPLES_DIR,
                                help='Data directory to validate')
    
    return parser.parse_args()


def command_setup(args):
    """Execute setup command"""
    from src.dataset_manager import setup_sample_dataset
    setup_sample_dataset(num_samples_per_class=args.samples)


def command_train(args):
    """Execute train command"""
    # Update config with arguments
    import config.config as cfg
    cfg.DEFAULT_MODEL = args.model
    cfg.EPOCHS = args.epochs
    cfg.BATCH_SIZE = args.batch_size
    
    # Run training
    import train_model
    train_model.main()


def command_predict(args):
    """Execute predict command"""
    if not args.image:
        print("Error: --image argument required")
        return
    
    from src.predictor import DefectPredictor
    from src.visualization import DefectVisualizer
    
    predictor = DefectPredictor(args.model)
    visualizer = DefectVisualizer() if args.visualize else None
    
    result = predictor.predict_with_visualization(args.image, visualizer) \
             if args.visualize else predictor.predict_single(args.image)
    
    predictor.print_prediction_result(result)


def command_evaluate(args):
    """Execute evaluate command"""
    from src.model import load_trained_model
    from src.data_preprocessing import DataPreprocessor
    from src.evaluation import ModelEvaluator
    
    # Load model
    model = load_trained_model(args.model)
    
    # Load test data
    preprocessor = DataPreprocessor()
    X, y, _ = preprocessor.load_dataset_from_directory(
        os.path.join(DATA_DIR, 'samples')
    )
    
    if len(X) == 0:
        print("No test data found")
        return
    
    # Evaluate
    y_cat = preprocessor.prepare_labels(y)
    evaluator = ModelEvaluator(model)
    evaluator.evaluate(X, y_cat)


def command_validate(args):
    """Execute validate command"""
    from src.dataset_manager import DatasetManager
    
    manager = DatasetManager()
    results = manager.validate_dataset(args.data_dir)
    manager.print_validation_results(results)


def main():
    """
    Main application entry point
    """
    args = parse_arguments()
    
    if args.command == 'setup':
        command_setup(args)
    elif args.command == 'train':
        command_train(args)
    elif args.command == 'predict':
        command_predict(args)
    elif args.command == 'evaluate':
        command_evaluate(args)
    elif args.command == 'validate':
        command_validate(args)
    else:
        print("Please specify a command. Use --help for options.")


if __name__ == "__main__":
    main()
