"""
Prediction Script for Defect Detection System

Standalone script to make predictions on new images
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import glob
from config.config import *
from src.predictor import DefectPredictor
from src.visualization import DefectVisualizer
from src.utils import print_section_header


def parse_arguments():
    """
    Parse command line arguments
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Defect Detection Prediction Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict single image
  python predict.py --image path/to/image.jpg --model models/best_model.h5
  
  # Predict multiple images
  python predict.py --images path/to/images/*.jpg --model models/best_model.h5
  
  # Predict with visualization
  python predict.py --image path/to/image.jpg --model models/best_model.h5 --visualize
  
  # Batch predict and save results
  python predict.py --images path/to/images/*.jpg --model models/best_model.h5 --output results.json
        """
    )
    
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--images', type=str, nargs='+', help='Paths to multiple images (supports wildcards)')
    parser.add_argument('--model', type=str, default=CHECKPOINT_PATH, 
                       help=f'Path to trained model (default: {CHECKPOINT_PATH})')
    parser.add_argument('--visualize', action='store_true', 
                       help='Create visualization of predictions')
    parser.add_argument('--output', type=str, help='Path to save results (JSON format)')
    parser.add_argument('--threshold', type=float, default=CONFIDENCE_THRESHOLD,
                       help=f'Confidence threshold for defect detection (default: {CONFIDENCE_THRESHOLD})')
    
    return parser.parse_args()


def main():
    """
    Main prediction function
    """
    args = parse_arguments()
    
    # Validate arguments
    if not args.image and not args.images:
        print("Error: Please provide --image or --images argument")
        return
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    print_section_header("DEFECT DETECTION SYSTEM - PREDICTION")
    
    # Initialize predictor
    predictor = DefectPredictor(args.model)
    
    # Initialize visualizer if needed
    visualizer = DefectVisualizer() if args.visualize else None
    
    # Single image prediction
    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: Image file not found: {args.image}")
            return
        
        result = predictor.predict_with_visualization(args.image, visualizer) \
                 if args.visualize else predictor.predict_single(args.image)
        
        predictor.print_prediction_result(result)
        
        # Check if defective
        is_defective = predictor.is_defective(result, args.threshold)
        
        print(f"\nDefect Detection Result:")
        print(f"  Product Status: {'DEFECTIVE' if is_defective else 'GOOD'}")
        print(f"  Threshold: {args.threshold:.2%}\n")
        
        # Save result if output specified
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=4)
            print(f"Result saved to {args.output}")
    
    # Batch prediction
    elif args.images:
        # Expand wildcards
        image_paths = []
        for pattern in args.images:
            image_paths.extend(glob.glob(pattern))
        
        if len(image_paths) == 0:
            print("Error: No images found matching the pattern")
            return
        
        print(f"Found {len(image_paths)} images to process\n")
        
        results = predictor.predict_batch(image_paths)
        
        # Print summary
        print_section_header("Batch Prediction Summary")
        
        defective_count = sum(1 for r in results 
                             if 'error' not in r and predictor.is_defective(r, args.threshold))
        
        print(f"Total Images: {len(results)}")
        print(f"Defective: {defective_count}")
        print(f"Good: {len(results) - defective_count}")
        print(f"Defect Rate: {defective_count / len(results):.2%}\n")
        
        # Save results if output specified
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
