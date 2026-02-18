"""
Visualization module for model evaluation and interpretation
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
import logging

logger = logging.getLogger(__name__)


class ResultVisualizer:
    """
    Visualization utilities for model results
    """
    
    def __init__(self, config, class_names):
        """
        Initialize visualizer
        
        Args:
            config (dict): Configuration dictionary
            class_names (list): List of class names
        """
        self.config = config
        self.class_names = class_names
        self.plots_dir = config['paths']['plots_dir']
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def plot_training_history(self, history, stage='stage1', save=True):
        """
        Plot training history (loss and metrics)
        
        Args:
            history (dict): Training history
            stage (str): Training stage
            save (bool): Whether to save plot
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training History - {stage.upper()}', fontsize=16, fontweight='bold')
        
        # Loss
        axes[0, 0].plot(history['loss'], label='Training Loss', linewidth=2)
        axes[0, 0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Model Loss', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[0, 1].plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0, 1].set_title('Model Accuracy', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision
        if 'precision' in history:
            axes[1, 0].plot(history['precision'], label='Training Precision', linewidth=2)
            axes[1, 0].plot(history['val_precision'], label='Validation Precision', linewidth=2)
            axes[1, 0].set_title('Model Precision', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Recall
        if 'recall' in history:
            axes[1, 1].plot(history['recall'], label='Training Recall', linewidth=2)
            axes[1, 1].plot(history['val_recall'], label='Validation Recall', linewidth=2)
            axes[1, 1].set_title('Model Recall', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.plots_dir, f'training_history_{stage}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to: {save_path}")
        
        return fig
    
    def plot_confusion_matrix(self, cm, normalize=False, save=True):
        """
        Plot confusion matrix
        
        Args:
            cm (np.array): Confusion matrix
            normalize (bool): Whether to normalize
            save (bool): Whether to save plot
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count' if not normalize else 'Proportion'},
            ax=ax
        )
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        
        # Rotate labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        plt.setp(ax.get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        
        if save:
            suffix = '_normalized' if normalize else ''
            save_path = os.path.join(self.plots_dir, f'confusion_matrix{suffix}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix plot saved to: {save_path}")
        
        return fig
    
    def plot_roc_curves(self, y_true_onehot, y_pred_probs, save=True):
        """
        Plot ROC curves for each class
        
        Args:
            y_true_onehot (np.array): True labels (one-hot)
            y_pred_probs (np.array): Prediction probabilities
            save (bool): Whether to save plot
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        from sklearn.metrics import roc_curve, auc
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot ROC curve for each class
        for i, class_name in enumerate(self.class_names):
            fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_pred_probs[:, i])
            roc_auc = auc(fpr, tpr)
            
            ax.plot(
                fpr, tpr,
                label=f'{class_name} (AUC = {roc_auc:.3f})',
                linewidth=2
            )
        
        # Plot diagonal
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.plots_dir, 'roc_curves.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curves plot saved to: {save_path}")
        
        return fig
    
    def plot_class_distribution(self, y_data, split_name='Dataset', save=True):
        """
        Plot class distribution
        
        Args:
            y_data (np.array): Labels
            split_name (str): Name of data split
            save (bool): Whether to save plot
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        if len(y_data.shape) > 1:
            y_indices = np.argmax(y_data, axis=1)
        else:
            y_indices = y_data
        
        # Count samples per class
        unique, counts = np.unique(y_indices, return_counts=True)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars = ax.bar(range(len(self.class_names)), 
                      [counts[i] if i in unique else 0 for i in range(len(self.class_names))],
                      color='steelblue', edgecolor='black', linewidth=1.5)
        
        ax.set_xlabel('Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
        ax.set_title(f'Class Distribution - {split_name}', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(self.class_names)))
        ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.plots_dir, f'class_distribution_{split_name.lower().replace(" ", "_")}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Class distribution plot saved to: {save_path}")
        
        return fig
    
    def plot_sample_predictions(self, images, y_true, y_pred, y_pred_probs, 
                                num_samples=10, save=True):
        """
        Plot sample predictions
        
        Args:
            images (np.array): Images
            y_true (np.array): True labels
            y_pred (np.array): Predicted labels
            y_pred_probs (np.array): Prediction probabilities
            num_samples (int): Number of samples to plot
            save (bool): Whether to save plot
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        num_samples = min(num_samples, len(images))
        
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.ravel()
        
        # Randomly select samples
        indices = np.random.choice(len(images), num_samples, replace=False)
        
        for i, idx in enumerate(indices):
            ax = axes[i]
            
            # Display image
            ax.imshow(images[idx])
            
            # Get prediction info
            true_class = self.class_names[y_true[idx]]
            pred_class = self.class_names[y_pred[idx]]
            confidence = y_pred_probs[idx][y_pred[idx]]
            
            # Set title color based on correctness
            color = 'green' if y_true[idx] == y_pred[idx] else 'red'
            
            ax.set_title(
                f'True: {true_class}\nPred: {pred_class}\nConf: {confidence:.2f}',
                fontsize=10,
                color=color,
                fontweight='bold'
            )
            ax.axis('off')
        
        plt.suptitle('Sample Predictions', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.plots_dir, 'sample_predictions.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Sample predictions plot saved to: {save_path}")
        
        return fig


class GradCAM:
    """
    Grad-CAM visualization for model interpretability
    """
    
    def __init__(self, model, layer_name=None):
        """
        Initialize Grad-CAM
        
        Args:
            model: Keras model
            layer_name (str): Name of layer to visualize (last conv layer if None)
        """
        self.model = model
        
        # Find last convolutional layer if not specified
        if layer_name is None:
            for layer in reversed(model.layers):
                if len(layer.output_shape) == 4:  # Conv layer
                    layer_name = layer.name
                    break
        
        self.layer_name = layer_name
        logger.info(f"Grad-CAM using layer: {layer_name}")
    
    def compute_heatmap(self, image, class_idx, eps=1e-8):
        """
        Compute Grad-CAM heatmap
        
        Args:
            image (np.array): Input image
            class_idx (int): Class index
            eps (float): Small epsilon for numerical stability
            
        Returns:
            np.array: Heatmap
        """
        # Ensure image has batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Create gradient model
        grad_model = keras.Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layer_name).output, self.model.output]
        )
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image)
            loss = predictions[:, class_idx]
        
        # Get gradients of loss w.r.t. conv layer output
        grads = tape.gradient(loss, conv_outputs)
        
        # Compute guided gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight conv outputs by gradients
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + eps)
        heatmap = heatmap.numpy()
        
        return heatmap
    
    def overlay_heatmap(self, image, heatmap, alpha=0.4, colormap='jet'):
        """
        Overlay heatmap on image
        
        Args:
            image (np.array): Original image
            heatmap (np.array): Heatmap
            alpha (float): Transparency
            colormap (str): Colormap name
            
        Returns:
            np.array: Overlayed image
        """
        # Resize heatmap to match image size
        heatmap = tf.image.resize(heatmap[..., np.newaxis], image.shape[:2])
        heatmap = heatmap.numpy().squeeze()
        
        # Apply colormap
        cmap = plt.get_cmap(colormap)
        heatmap_colored = cmap(heatmap)[..., :3]
        
        # Overlay
        overlayed = heatmap_colored * alpha + image * (1 - alpha)
        overlayed = np.clip(overlayed, 0, 1)
        
        return overlayed


if __name__ == "__main__":
    from utils import setup_logging
    
    setup_logging()
    logger.info("Visualization module loaded successfully!")