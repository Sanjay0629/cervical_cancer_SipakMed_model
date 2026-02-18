#!/usr/bin/env python3
"""
Main script for training cervical cytology classification model
"""

import os
import sys
import argparse

# Add parent directory and src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, 'src')
sys.path.insert(0, parent_dir)
sys.path.insert(0, src_dir)

from src.train import ModelTrainer
from src.utils import (
    setup_logging, load_config, set_seeds, 
    create_directories, get_gpu_info
)
from src.visualization import ResultVisualizer
import logging

logger = logging.getLogger(__name__)


def main(config_path=None, resume=False):
    """
    Main training function
    
    Args:
        config_path (str): Path to configuration file
        resume (bool): Whether to resume training from checkpoint
    """
    # Setup logging
    setup_logging()
    
    logger.info("="*60)
    logger.info("CERVICAL CYTOLOGY CLASSIFICATION - TRAINING")
    logger.info("="*60)
    
    # Default config path
    if config_path is None:
        config_path = os.path.join(parent_dir, "config", "config.yaml")
    
    # Load configuration
    logger.info(f"Loading configuration from: {config_path}")
    config = load_config(config_path)
    
    # Set seeds for reproducibility
    set_seeds(config['dataset']['seed'])
    logger.info(f"Random seed set to: {config['dataset']['seed']}")
    
    # Create directories
    create_directories(config)
    logger.info("Project directories created")
    
    # Get GPU info
    gpu_info = get_gpu_info()
    logger.info(f"GPU Available: {gpu_info['gpu_available']}")
    if gpu_info['gpu_available']:
        logger.info(f"Number of GPUs: {gpu_info['num_gpus']}")
    
    # Initialize trainer
    logger.info("Initializing model trainer...")
    trainer = ModelTrainer(config)
    
    # Run training pipeline
    try:
        logger.info("\nStarting training pipeline...")
        results = trainer.train_complete_pipeline()
        
        # Visualize training history
        logger.info("\nGenerating training visualizations...")
        visualizer = ResultVisualizer(config, results['data_info']['class_names'])
        
        # Plot Stage 1 history
        if 'history_stage1' in results:
            visualizer.plot_training_history(
                results['history_stage1'],
                stage='stage1',
                save=True
            )
        
        # Plot Stage 2 history
        if 'history_stage2' in results:
            visualizer.plot_training_history(
                results['history_stage2'],
                stage='stage2',
                save=True
            )
        
        # Success message
        logger.info("\n" + "="*60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info(f"\nModel saved to: {results['model_path']}")
        logger.info(f"Best model: {results['best_model_path']}")
        logger.info(f"\nDataset Statistics:")
        logger.info(f"  Training samples:   {results['data_info']['num_train']}")
        logger.info(f"  Validation samples: {results['data_info']['num_val']}")
        logger.info(f"  Test samples:       {results['data_info']['num_test']}")
        logger.info(f"  Number of classes:  {results['data_info']['num_classes']}")
        logger.info(f"\nClasses: {', '.join(results['data_info']['class_names'])}")
        logger.info("\n" + "="*60)
        logger.info("\nNext steps:")
        logger.info("1. Run 'python scripts/evaluate_model.py' to evaluate the model")
        logger.info("2. Run 'python api/app.py' to start the API server")
        logger.info("3. Check results/ directory for training plots and logs")
        logger.info("="*60 + "\n")
        
        return results
        
    except KeyboardInterrupt:
        logger.warning("\n\nTraining interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"\n\nTraining failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train cervical cytology classification model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from checkpoint"
    )
    
    args = parser.parse_args()
    
    main(config_path=args.config, resume=args.resume)