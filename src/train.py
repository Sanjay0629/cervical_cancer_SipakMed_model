"""
Training module for cervical cytology classification
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import logging
from datetime import datetime

try:
    from src.data_loader import CervicalDataLoader
    from src.model import CervicalCNNModel, compile_model, create_callbacks
    from src.utils import (
        load_config, set_seeds, create_directories, 
        compute_class_weights, print_class_distribution,
        save_model_info, get_gpu_info
    )
    from src.preprocessing import get_augmentation_layer
except ImportError:
    from data_loader import CervicalDataLoader
    from model import CervicalCNNModel, compile_model, create_callbacks
    from utils import (
        load_config, set_seeds, create_directories, 
        compute_class_weights, print_class_distribution,
        save_model_info, get_gpu_info
    )
    from preprocessing import get_augmentation_layer

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Model training pipeline
    """
    
    def __init__(self, config):
        """
        Initialize trainer
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        self.model = None
        self.history = None
        self.data = None
        
    def load_and_prepare_data(self):
        """
        Load and prepare dataset
        """
        logger.info("Loading and preparing dataset...")
        
        # Initialize data loader
        data_loader = CervicalDataLoader(self.config)
        
        # Prepare dataset
        self.data = data_loader.prepare_dataset()
        
        # Print class distribution
        print_class_distribution(
            self.data['y_train'],
            self.data['class_names'],
            "Training Set"
        )
        print_class_distribution(
            self.data['y_val'],
            self.data['class_names'],
            "Validation Set"
        )
        print_class_distribution(
            self.data['y_test'],
            self.data['class_names'],
            "Test Set"
        )
        
        logger.info("Dataset prepared successfully")
    
    def create_data_generators(self, stage='stage1'):
        """
        Create data generators with augmentation
        
        Args:
            stage (str): Training stage
            
        Returns:
            tuple: (train_dataset, val_dataset)
        """
        batch_size = self.config['training'][stage]['batch_size']
        
        # Get augmentation layer
        aug_layer = get_augmentation_layer(self.config['augmentation'])
        
        # Training dataset with augmentation
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (self.data['X_train'], self.data['y_train'])
        )
        train_dataset = train_dataset.shuffle(buffer_size=len(self.data['X_train']))
        train_dataset = train_dataset.map(
            lambda x, y: (aug_layer(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        train_dataset = train_dataset.batch(batch_size)
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        
        # Validation dataset without augmentation
        val_dataset = tf.data.Dataset.from_tensor_slices(
            (self.data['X_val'], self.data['y_val'])
        )
        val_dataset = val_dataset.batch(batch_size)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
        
        logger.info(f"Data generators created for {stage}")
        logger.info(f"Batch size: {batch_size}")
        
        return train_dataset, val_dataset
    
    def train_stage1(self):
        """
        Stage 1: Train custom head with frozen base model
        
        Returns:
            keras.callbacks.History: Training history
        """
        logger.info("="*60)
        logger.info("STAGE 1: Training custom head (base model frozen)")
        logger.info("="*60)
        
        # Build model
        model_builder = CervicalCNNModel(self.config)
        self.model = model_builder.build_model(freeze_base=True)
        
        # Compile model
        self.model = compile_model(self.model, self.config, stage='stage1')
        
        # Create data generators
        train_dataset, val_dataset = self.create_data_generators(stage='stage1')
        
        # Compute class weights
        class_weights = None
        if self.config['training'].get('use_class_weights', False):
            class_weights = compute_class_weights(
                self.data['y_train_int'],
                self.config['model']['num_classes']
            )
            logger.info(f"Class weights: {class_weights}")
        
        # Create callbacks
        callbacks = create_callbacks(self.config, stage='stage1')
        
        # Train model
        epochs = self.config['training']['stage1']['epochs']
        
        logger.info(f"Starting Stage 1 training for {epochs} epochs...")
        
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        logger.info("Stage 1 training completed")
        
        return history
    
    def train_stage2(self):
        """
        Stage 2: Fine-tune with partially unfrozen base model
        
        Returns:
            keras.callbacks.History: Training history
        """
        if self.model is None:
            raise ValueError("Model not initialized. Run train_stage1 first.")
        
        logger.info("="*60)
        logger.info("STAGE 2: Fine-tuning with unfrozen layers")
        logger.info("="*60)
        
        # Unfreeze last layers
        model_builder = CervicalCNNModel(self.config)
        num_layers_unfreeze = self.config['training']['stage2']['unfreeze_layers']
        model_builder.unfreeze_layers(self.model, num_layers_unfreeze)
        
        # Recompile model with lower learning rate
        self.model = compile_model(self.model, self.config, stage='stage2')
        
        # Create data generators
        train_dataset, val_dataset = self.create_data_generators(stage='stage2')
        
        # Compute class weights
        class_weights = None
        if self.config['training'].get('use_class_weights', False):
            class_weights = compute_class_weights(
                self.data['y_train_int'],
                self.config['model']['num_classes']
            )
        
        # Create callbacks
        callbacks = create_callbacks(self.config, stage='stage2')
        
        # Train model
        epochs = self.config['training']['stage2']['epochs']
        
        logger.info(f"Starting Stage 2 training for {epochs} epochs...")
        
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        logger.info("Stage 2 training completed")
        
        return history
    
    def save_model(self, filename=None):
        """
        Save trained model
        
        Args:
            filename (str): Model filename
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        # Create filename with timestamp
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cervical_model_{self.config['model']['base_model']}_{timestamp}.h5"
        
        # Save path
        save_path = os.path.join(self.config['paths']['models_dir'], filename)
        
        # Save model
        self.model.save(save_path)
        logger.info(f"Model saved to: {save_path}")
        
        # Save model info
        info_path = save_path.replace('.h5', '_info.json')
        save_model_info(
            self.model,
            self.config,
            info_path,
            additional_info={
                'training_completed': datetime.now().isoformat(),
                'class_names': self.data['class_names']
            }
        )
        logger.info(f"Model info saved to: {info_path}")
        
        return save_path
    
    def train_complete_pipeline(self):
        """
        Complete two-stage training pipeline
        
        Returns:
            dict: Training results
        """
        # Load data
        self.load_and_prepare_data()
        
        # Stage 1: Train head
        history_stage1 = self.train_stage1()
        
        # Stage 2: Fine-tune
        history_stage2 = self.train_stage2()
        
        # Save final model
        model_path = self.save_model()
        
        # Also save as best_model.h5 for easy access
        best_model_path = os.path.join(
            self.config['paths']['models_dir'],
            'best_model.h5'
        )
        self.model.save(best_model_path)
        logger.info(f"Best model also saved to: {best_model_path}")
        
        results = {
            'model_path': model_path,
            'best_model_path': best_model_path,
            'history_stage1': history_stage1.history,
            'history_stage2': history_stage2.history,
            'data_info': {
                'num_train': len(self.data['X_train']),
                'num_val': len(self.data['X_val']),
                'num_test': len(self.data['X_test']),
                'num_classes': self.data['num_classes'],
                'class_names': self.data['class_names']
            }
        }
        
        return results


def main():
    """
    Main training function
    """
    from utils import setup_logging
    
    # Setup logging
    logger_instance = setup_logging()
    
    # Load configuration
    config = load_config("config/config.yaml")
    
    # Set seeds for reproducibility
    set_seeds(config['dataset']['seed'])
    
    # Create directories
    create_directories(config)
    
    # Get GPU info
    gpu_info = get_gpu_info()
    logger.info(f"GPU Info: {gpu_info}")
    
    # Initialize trainer
    trainer = ModelTrainer(config)
    
    # Run complete training pipeline
    try:
        results = trainer.train_complete_pipeline()
        
        logger.info("="*60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        logger.info(f"Model saved to: {results['model_path']}")
        logger.info(f"Training data: {results['data_info']['num_train']} samples")
        logger.info(f"Validation data: {results['data_info']['num_val']} samples")
        logger.info(f"Test data: {results['data_info']['num_test']} samples")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()