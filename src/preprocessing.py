"""
Image preprocessing and data augmentation module
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import logging

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Image preprocessing utilities
    """
    
    def __init__(self, config):
        """
        Initialize preprocessor
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        self.target_size = tuple(config['preprocessing']['target_size'])
        
    @staticmethod
    def resize_image(image, target_size):
        """
        Resize image to target size
        
        Args:
            image (np.array): Input image
            target_size (tuple): Target size (height, width)
            
        Returns:
            np.array: Resized image
        """
        return cv2.resize(image, target_size[::-1], interpolation=cv2.INTER_AREA)
    
    @staticmethod
    def normalize_image(image):
        """
        Normalize image to [0, 1]
        
        Args:
            image (np.array): Input image
            
        Returns:
            np.array: Normalized image
        """
        return image.astype(np.float32) / 255.0
    
    @staticmethod
    def convert_to_rgb(image):
        """
        Convert image to RGB if needed
        
        Args:
            image (np.array): Input image
            
        Returns:
            np.array: RGB image
        """
        if len(image.shape) == 2:  # Grayscale
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        return image
    
    @staticmethod
    def stain_normalization(image, method='reinhard'):
        """
        Apply stain normalization (placeholder for future implementation)
        
        Args:
            image (np.array): Input image
            method (str): Normalization method
            
        Returns:
            np.array: Normalized image
        """
        # TODO: Implement stain normalization using staintools or custom method
        # This is a placeholder that returns the original image
        logger.warning("Stain normalization not yet implemented")
        return image
    
    def preprocess_single_image(self, image_path):
        """
        Preprocess a single image for inference
        
        Args:
            image_path (str): Path to image
            
        Returns:
            np.array: Preprocessed image
        """
        # Read image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image = self.resize_image(image, self.target_size)
        
        # Normalize
        image = self.normalize_image(image)
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image


def get_augmentation_layer(augmentation_config):
    """
    Create data augmentation layer using Keras preprocessing layers
    
    Args:
        augmentation_config (dict): Augmentation configuration
        
    Returns:
        keras.Sequential: Augmentation layer
    """
    aug_layers = []
    
    # Random rotation
    if augmentation_config.get('rotation_range', 0) > 0:
        aug_layers.append(
            layers.RandomRotation(
                factor=augmentation_config['rotation_range'] / 360.0,
                fill_mode=augmentation_config.get('fill_mode', 'nearest')
            )
        )
    
    # Random translation
    if augmentation_config.get('width_shift_range', 0) > 0:
        aug_layers.append(
            layers.RandomTranslation(
                height_factor=augmentation_config.get('height_shift_range', 0),
                width_factor=augmentation_config.get('width_shift_range', 0),
                fill_mode=augmentation_config.get('fill_mode', 'nearest')
            )
        )
    
    # Random zoom
    if augmentation_config.get('zoom_range', 0) > 0:
        zoom_range = augmentation_config['zoom_range']
        aug_layers.append(
            layers.RandomZoom(
                height_factor=(-zoom_range, zoom_range),
                width_factor=(-zoom_range, zoom_range),
                fill_mode=augmentation_config.get('fill_mode', 'nearest')
            )
        )
    
    # Random horizontal flip
    if augmentation_config.get('horizontal_flip', False):
        aug_layers.append(layers.RandomFlip(mode="horizontal"))
    
    # Random vertical flip
    if augmentation_config.get('vertical_flip', False):
        aug_layers.append(layers.RandomFlip(mode="vertical"))
    
    # Random brightness
    if augmentation_config.get('brightness_range'):
        brightness_range = augmentation_config['brightness_range']
        brightness_delta = (brightness_range[1] - brightness_range[0]) / 2
        aug_layers.append(
            layers.RandomBrightness(
                factor=brightness_delta
            )
        )
    
    # Random contrast (optional)
    if augmentation_config.get('contrast_range'):
        contrast_range = augmentation_config['contrast_range']
        aug_layers.append(
            layers.RandomContrast(factor=contrast_range)
        )
    
    # Combine all augmentation layers
    if aug_layers:
        return keras.Sequential(aug_layers, name="augmentation")
    else:
        return keras.Sequential([], name="no_augmentation")


class CustomAugmentation(keras.layers.Layer):
    """
    Custom augmentation layer with medical imaging specific augmentations
    """
    
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        
    def call(self, images, training=None):
        """
        Apply augmentation during training
        
        Args:
            images: Input images
            training: Whether in training mode
            
        Returns:
            Augmented images
        """
        if training:
            # Apply Gaussian noise (simulates imaging artifacts)
            if self.config.get('gaussian_noise', False):
                noise = tf.random.normal(
                    shape=tf.shape(images),
                    mean=0.0,
                    stddev=0.01,
                    dtype=images.dtype
                )
                images = images + noise
                images = tf.clip_by_value(images, 0.0, 1.0)
            
            # Apply Gaussian blur (simulates focus variation)
            if self.config.get('gaussian_blur', False):
                images = self.apply_gaussian_blur(images)
        
        return images
    
    @tf.function
    def apply_gaussian_blur(self, images):
        """
        Apply Gaussian blur to images
        
        Args:
            images: Input images
            
        Returns:
            Blurred images
        """
        # Simple box blur approximation
        kernel_size = 3
        kernel = tf.ones((kernel_size, kernel_size, 3, 3)) / (kernel_size ** 2)
        
        blurred = tf.nn.depthwise_conv2d(
            images,
            kernel,
            strides=[1, 1, 1, 1],
            padding='SAME'
        )
        
        # Randomly apply blur to some images
        mask = tf.random.uniform(tf.shape(images)[:1]) > 0.5
        mask = tf.reshape(mask, [-1, 1, 1, 1])
        mask = tf.cast(mask, images.dtype)
        
        return images * (1 - mask) + blurred * mask


def create_preprocessing_model(config, include_augmentation=False):
    """
    Create preprocessing model that can be included in the main model
    
    Args:
        config (dict): Configuration dictionary
        include_augmentation (bool): Whether to include augmentation
        
    Returns:
        keras.Model: Preprocessing model
    """
    inputs = keras.Input(shape=config['model']['input_shape'])
    x = inputs
    
    # Augmentation (only if specified)
    if include_augmentation:
        aug_layer = get_augmentation_layer(config['augmentation'])
        x = aug_layer(x)
        
        # Custom augmentation
        if config['augmentation'].get('gaussian_noise') or config['augmentation'].get('gaussian_blur'):
            custom_aug = CustomAugmentation(config['augmentation'])
            x = custom_aug(x)
    
    # Preprocessing for specific models (if needed)
    model_name = config['model']['base_model']
    
    if model_name == 'ResNet50':
        x = tf.keras.applications.resnet50.preprocess_input(x)
    elif model_name == 'MobileNetV2':
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    elif model_name == 'EfficientNetB0':
        x = tf.keras.applications.efficientnet.preprocess_input(x)
    elif model_name == 'DenseNet121':
        x = tf.keras.applications.densenet.preprocess_input(x)
    
    model = keras.Model(inputs, x, name="preprocessing")
    return model


def visualize_augmentation(images, labels, augmentation_config, num_samples=5):
    """
    Visualize augmentation effects
    
    Args:
        images (np.array): Original images
        labels (np.array): Labels
        augmentation_config (dict): Augmentation configuration
        num_samples (int): Number of samples to visualize
        
    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt
    
    aug_layer = get_augmentation_layer(augmentation_config)
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 5))
    
    for i in range(num_samples):
        # Original image
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title(f"Original - Class: {np.argmax(labels[i])}")
        axes[i, 0].axis('off')
        
        # Augmented image
        augmented = aug_layer(tf.expand_dims(images[i], 0), training=True)
        axes[i, 1].imshow(augmented[0].numpy())
        axes[i, 1].set_title(f"Augmented - Class: {np.argmax(labels[i])}")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Test preprocessing
    import yaml
    from utils import setup_logging
    
    setup_logging()
    
    # Load config
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Test augmentation layer
    aug_layer = get_augmentation_layer(config['augmentation'])
    print(f"Augmentation layer created: {aug_layer.name}")
    print(f"Number of layers: {len(aug_layer.layers)}")
    
    # Test on dummy image
    dummy_image = tf.random.uniform((1, 224, 224, 3))
    augmented = aug_layer(dummy_image, training=True)
    print(f"Input shape: {dummy_image.shape}")
    print(f"Output shape: {augmented.shape}")
    
    print("\nPreprocessing module loaded successfully!")