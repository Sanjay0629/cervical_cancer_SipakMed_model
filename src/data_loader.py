"""
Data loading and splitting module for cervical cytology dataset
"""

import os
import sys
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class CervicalDataLoader:
    """
    Data loader for cervical cytology images
    """
    
    def __init__(self, config):
        """
        Initialize data loader
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        self.data_dir = config['dataset']['data_dir']
        self.classes = config['dataset']['classes']
        self.target_size = tuple(config['preprocessing']['target_size'])
        self.seed = config['dataset']['seed']
        
        self.num_classes = len(self.classes)
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
    def load_data(self):
        """
        Load all images and labels from directory structure
        
        Returns:
            tuple: (images, labels, file_paths)
        """
        logger.info("Loading dataset from: %s", self.data_dir)
        
        images = []
        labels = []
        file_paths = []
        
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            
            if not os.path.exists(class_dir):
                logger.warning(f"Directory not found: {class_dir}")
                continue
            
            class_idx = self.class_to_idx[class_name]
            
            # Get all .bmp files
            image_files = list(Path(class_dir).glob("*.bmp"))
            
            logger.info(f"Found {len(image_files)} images in {class_name}")
            
            for img_path in image_files:
                try:
                    # Load image
                    img = Image.open(img_path)
                    
                    # Convert to RGB if needed
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Resize
                    img = img.resize(self.target_size)
                    
                    # Convert to array
                    img_array = np.array(img)
                    
                    images.append(img_array)
                    labels.append(class_idx)
                    file_paths.append(str(img_path))
                    
                except Exception as e:
                    logger.error(f"Error loading image {img_path}: {e}")
                    continue
        
        images = np.array(images, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)
        
        logger.info(f"Total images loaded: {len(images)}")
        logger.info(f"Image shape: {images.shape}")
        
        return images, labels, file_paths
    
    def normalize_images(self, images):
        """
        Normalize pixel values to [0, 1]
        
        Args:
            images (np.array): Image array
            
        Returns:
            np.array: Normalized images
        """
        return images / 255.0
    
    def split_data(self, images, labels, file_paths=None):
        """
        Split data into train, validation, and test sets with stratification
        
        Args:
            images (np.array): Image array
            labels (np.array): Label array
            file_paths (list): List of file paths (optional)
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        train_split = self.config['dataset']['train_split']
        val_split = self.config['dataset']['val_split']
        test_split = self.config['dataset']['test_split']
        
        # Ensure splits sum to 1.0
        assert abs(train_split + val_split + test_split - 1.0) < 1e-5, \
            "Train, validation, and test splits must sum to 1.0"
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, labels,
            test_size=test_split,
            random_state=self.seed,
            stratify=labels
        )
        
        # Second split: separate train and validation
        val_size_adjusted = val_split / (train_split + val_split)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=self.seed,
            stratify=y_temp
        )
        
        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Validation set: {len(X_val)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def convert_labels_to_categorical(self, labels):
        """
        Convert integer labels to one-hot encoded format
        
        Args:
            labels (np.array): Integer labels
            
        Returns:
            np.array: One-hot encoded labels
        """
        return tf.keras.utils.to_categorical(labels, num_classes=self.num_classes)
    
    def prepare_dataset(self):
        """
        Complete pipeline: load, normalize, split data
        
        Returns:
            dict: Dictionary containing all data splits
        """
        # Load data
        images, labels, file_paths = self.load_data()
        
        # Normalize images
        images = self.normalize_images(images)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(
            images, labels, file_paths
        )
        
        # Convert labels to categorical
        y_train_cat = self.convert_labels_to_categorical(y_train)
        y_val_cat = self.convert_labels_to_categorical(y_val)
        y_test_cat = self.convert_labels_to_categorical(y_test)
        
        # Create result dictionary
        data = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train_cat,
            'y_val': y_val_cat,
            'y_test': y_test_cat,
            'y_train_int': y_train,  # Keep integer labels for class weights
            'y_val_int': y_val,
            'y_test_int': y_test,
            'class_names': self.classes,
            'num_classes': self.num_classes
        }
        
        return data
    
    def get_class_distribution(self, labels):
        """
        Get class distribution statistics
        
        Args:
            labels (np.array): Labels array
            
        Returns:
            dict: Class distribution
        """
        unique, counts = np.unique(labels, return_counts=True)
        distribution = {self.classes[i]: count for i, count in zip(unique, counts)}
        return distribution


def create_tf_dataset(X, y, batch_size, shuffle=True, augment=False, augmentation_config=None):
    """
    Create TensorFlow dataset with optional augmentation
    
    Args:
        X (np.array): Images
        y (np.array): Labels
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle data
        augment (bool): Whether to apply augmentation
        augmentation_config (dict): Augmentation configuration
        
    Returns:
        tf.data.Dataset: TensorFlow dataset
    """
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))
    
    if augment and augmentation_config:
        from src.preprocessing import get_augmentation_layer
        aug_layer = get_augmentation_layer(augmentation_config)
        dataset = dataset.map(
            lambda x, y: (aug_layer(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset