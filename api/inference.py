"""
Inference module for cervical cytology classification
"""

import os
import numpy as np
import tensorflow as tf
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class CervicalInference:
    """
    Inference engine for cervical cytology classification
    """
    
    def __init__(self, config):
        """
        Initialize inference engine
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        self.model = None
        self.class_names = config['dataset']['classes']
        self.num_classes = len(self.class_names)
        self.target_size = tuple(config['preprocessing']['target_size'])
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """Load trained model"""
        model_path = self.config['api']['model_path']
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")
        
        logger.info(f"Loading model from: {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        logger.info("Model loaded successfully")
    
    def preprocess_image(self, image):
        """
        Preprocess image for prediction
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            np.array: Preprocessed image
        """
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to RGB if needed
        if len(image.shape) == 2:  # Grayscale
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:  # RGBA
            image = image[:, :, :3]
        
        # Resize
        image = tf.image.resize(image, self.target_size)
        
        # Normalize
        image = image / 255.0
        
        # Add batch dimension
        image = tf.expand_dims(image, axis=0)
        
        return image
    
    def predict_image(self, image):
        """
        Predict class for a single image
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            dict: Prediction results
        """
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Make prediction
        predictions = self.model.predict(processed_image, verbose=0)
        probabilities = predictions[0]
        
        # Get predicted class
        predicted_idx = np.argmax(probabilities)
        predicted_class = self.class_names[predicted_idx]
        confidence = float(probabilities[predicted_idx])
        
        # Get all probabilities
        all_probs = {
            class_name: float(prob)
            for class_name, prob in zip(self.class_names, probabilities)
        }
        
        # Get top 3 predictions
        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        top_3_predictions = [
            {
                'class': self.class_names[idx],
                'probability': float(probabilities[idx])
            }
            for idx in top_3_indices
        ]
        
        result = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'predicted_index': int(predicted_idx),
            'all_probabilities': all_probs,
            'top_3_predictions': top_3_predictions
        }
        
        return result
    
    def predict_batch(self, images):
        """
        Predict classes for multiple images
        
        Args:
            images: List of PIL Images or numpy arrays
            
        Returns:
            list: List of prediction results
        """
        results = []
        
        for image in images:
            try:
                result = self.predict_image(image)
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting image: {e}")
                results.append({
                    'error': str(e),
                    'predicted_class': None,
                    'confidence': 0.0
                })
        
        return results
    
    def predict_from_path(self, image_path):
        """
        Predict class from image file path
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            dict: Prediction results
        """
        # Load image
        image = Image.open(image_path)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Predict
        result = self.predict_image(image)
        result['image_path'] = image_path
        
        return result


def main():
    """
    Test inference module
    """
    import yaml
    from utils import setup_logging
    
    setup_logging()
    
    # Load config
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize inference engine
    try:
        inference = CervicalInference(config)
        logger.info("Inference engine initialized successfully")
        
        # Test with dummy image
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        dummy_pil = Image.fromarray(dummy_image)
        
        result = inference.predict_image(dummy_pil)
        
        print("\n" + "="*60)
        print("TEST PREDICTION RESULTS")
        print("="*60)
        print(f"Predicted Class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print("\nAll Probabilities:")
        for class_name, prob in result['all_probabilities'].items():
            print(f"  {class_name}: {prob:.4f}")
        print("="*60)
        
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.error("Please train the model first using train.py")


if __name__ == "__main__":
    main()