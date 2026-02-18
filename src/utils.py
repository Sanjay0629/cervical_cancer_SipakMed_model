"""
Utility functions for cervical cytology classification
"""

import os
import yaml
import json
import random
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
import logging


def setup_logging(log_dir="results/logs", log_file="training.log"):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, log_file)),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def set_seeds(seed=42):
    """
    Set random seeds for reproducibility
    
    Args:
        seed (int): Random seed value
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # For deterministic behavior (may reduce performance)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'


def load_config(config_path="config/config.yaml"):
    """
    Load configuration from YAML file
    
    Args:
        config_path (str): Path to config file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config, save_path):
    """
    Save configuration to file
    
    Args:
        config (dict): Configuration dictionary
        save_path (str): Path to save config
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def create_directories(config):
    """
    Create necessary directories for the project
    
    Args:
        config (dict): Configuration dictionary
    """
    directories = [
        config['paths']['models_dir'],
        config['paths']['checkpoints_dir'],
        config['paths']['results_dir'],
        config['paths']['logs_dir'],
        config['paths']['plots_dir'],
        config['paths']['reports_dir'],
        os.path.join(config['paths']['results_dir'], 'predictions')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def get_timestamp():
    """
    Get current timestamp string
    
    Returns:
        str: Timestamp in format YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_model_info(model, config, save_path, additional_info=None):
    """
    Save model metadata and training information
    
    Args:
        model: Keras model
        config (dict): Configuration dictionary
        save_path (str): Path to save model info
        additional_info (dict): Additional information to save
    """
    info = {
        'timestamp': get_timestamp(),
        'model_name': config['model']['base_model'],
        'input_shape': config['model']['input_shape'],
        'num_classes': config['model']['num_classes'],
        'total_parameters': int(model.count_params()),
        'trainable_parameters': int(sum([tf.size(w).numpy() for w in model.trainable_weights])),
        'config': config
    }
    
    if additional_info:
        info.update(additional_info)
    
    with open(save_path, 'w') as f:
        json.dump(info, f, indent=4)


def load_model_with_info(model_path):
    """
    Load saved model and its metadata
    
    Args:
        model_path (str): Path to saved model
        
    Returns:
        tuple: (model, info_dict)
    """
    model = tf.keras.models.load_model(model_path)
    
    info_path = model_path.replace('.h5', '_info.json').replace('.keras', '_info.json')
    
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            info = json.load(f)
    else:
        info = {}
    
    return model, info


def compute_class_weights(y_train, num_classes):
    """
    Compute class weights for imbalanced datasets
    
    Args:
        y_train (np.array): Training labels (one-hot encoded)
        num_classes (int): Number of classes
        
    Returns:
        dict: Class weights dictionary
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    # Convert one-hot to class indices
    if len(y_train.shape) > 1:
        y_indices = np.argmax(y_train, axis=1)
    else:
        y_indices = y_train
    
    # Compute class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_indices),
        y=y_indices
    )
    
    # Convert to dictionary
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    return class_weight_dict


def print_class_distribution(y_data, class_names, split_name="Dataset"):
    """
    Print class distribution statistics
    
    Args:
        y_data (np.array): Labels (one-hot encoded)
        class_names (list): List of class names
        split_name (str): Name of the data split
    """
    if len(y_data.shape) > 1:
        y_indices = np.argmax(y_data, axis=1)
    else:
        y_indices = y_data
    
    print(f"\n{split_name} Class Distribution:")
    print("-" * 50)
    
    for i, class_name in enumerate(class_names):
        count = np.sum(y_indices == i)
        percentage = (count / len(y_indices)) * 100
        print(f"{class_name:30s}: {count:5d} ({percentage:5.2f}%)")
    
    print("-" * 50)
    print(f"Total samples: {len(y_indices)}\n")


def get_gpu_info():
    """
    Get GPU information
    
    Returns:
        dict: GPU information
    """
    gpus = tf.config.list_physical_devices('GPU')
    
    info = {
        'num_gpus': len(gpus),
        'gpu_available': len(gpus) > 0,
        'gpu_names': [gpu.name for gpu in gpus]
    }
    
    if info['gpu_available']:
        for gpu in gpus:
            print(f"GPU: {gpu.name}")
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"Memory growth setting failed: {e}")
    else:
        print("No GPU found. Training will use CPU.")
    
    return info


def ensure_directory(path):
    """
    Ensure directory exists, create if it doesn't
    
    Args:
        path (str): Directory path
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def get_latest_model(models_dir):
    """
    Get the latest saved model from directory
    
    Args:
        models_dir (str): Directory containing saved models
        
    Returns:
        str: Path to latest model file
    """
    model_files = list(Path(models_dir).glob("*.h5")) + list(Path(models_dir).glob("*.keras"))
    
    if not model_files:
        return None
    
    latest_model = max(model_files, key=os.path.getctime)
    return str(latest_model)


class CustomMetrics:
    """Custom metrics for medical image classification"""
    
    @staticmethod
    def specificity(y_true, y_pred):
        """Calculate specificity metric"""
        true_negatives = tf.reduce_sum(tf.cast((1 - y_true) * (1 - y_pred), tf.float32))
        possible_negatives = tf.reduce_sum(tf.cast(1 - y_true, tf.float32))
        return true_negatives / (possible_negatives + tf.keras.backend.epsilon())
    
    @staticmethod
    def sensitivity(y_true, y_pred):
        """Calculate sensitivity (recall) metric"""
        true_positives = tf.reduce_sum(tf.cast(y_true * y_pred, tf.float32))
        possible_positives = tf.reduce_sum(tf.cast(y_true, tf.float32))
        return true_positives / (possible_positives + tf.keras.backend.epsilon())


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    # Test seed setting
    set_seeds(42)
    print("✓ Seeds set")
    
    # Test GPU info
    gpu_info = get_gpu_info()
    print(f"✓ GPU Info: {gpu_info}")
    
    # Test timestamp
    timestamp = get_timestamp()
    print(f"✓ Timestamp: {timestamp}")
    
    print("\nAll utility functions loaded successfully!")