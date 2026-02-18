#!/usr/bin/env python3
"""
Script for making predictions on single images
"""

import os
import sys
import argparse
from PIL import Image

# Add parent directory and src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, 'src')
api_dir = os.path.join(parent_dir, 'api')
sys.path.insert(0, parent_dir)
sys.path.insert(0, src_dir)
sys.path.insert(0, api_dir)

from api.inference import CervicalInference
from src.utils import setup_logging, load_config
import logging

logger = logging.getLogger(__name__)


def main(image_path, config_path=None, show_all=False):
    """
    Predict cell type for a single image
    
    Args:
        image_path (str): Path to image file
        config_path (str): Path to configuration file
        show_all (bool): Show all class probabilities
    """
    # Setup logging
    setup_logging()
    
    # Default config path
    if config_path is None:
        config_path = os.path.join(parent_dir, "config", "config.yaml")
    
    # Load configuration
    config = load_config(config_path)
    
    # Check if image exists
    if not os.path.exists(image_path):
        logger.error(f"Image not found: {image_path}")
        sys.exit(1)
    
    try:
        # Initialize inference engine
        logger.info("Loading model...")
        inference = CervicalInference(config)
        logger.info("Model loaded successfully")
        
        # Make prediction
        logger.info(f"\nPredicting image: {image_path}")
        result = inference.predict_from_path(image_path)
        
        # Print results
        print("\n" + "="*60)
        print("PREDICTION RESULTS")
        print("="*60)
        print(f"\nImage: {os.path.basename(image_path)}")
        print(f"\nPredicted Class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.2%}")
        
        print("\nTop 3 Predictions:")
        print("-"*60)
        for i, pred in enumerate(result['top_3_predictions'], 1):
            class_name = pred['class']
            prob = pred['probability']
            print(f"{i}. {class_name:<30} {prob:.2%}")
        
        if show_all:
            print("\nAll Class Probabilities:")
            print("-"*60)
            for class_name, prob in sorted(
                result['all_probabilities'].items(),
                key=lambda x: x[1],
                reverse=True
            ):
                print(f"{class_name:<30} {prob:.2%}")
        
        print("="*60 + "\n")
        
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.error("Please train the model first using: python scripts/train_model.py")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict cervical cell type from image"
    )
    parser.add_argument(
        "image",
        type=str,
        help="Path to image file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show probabilities for all classes"
    )
    
    args = parser.parse_args()
    
    main(image_path=args.image, config_path=args.config, show_all=args.all)