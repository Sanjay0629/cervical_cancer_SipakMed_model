#!/usr/bin/env python3
"""
Main script for evaluating cervical cytology classification model
"""

import os
import sys
import argparse
import numpy as np

# Add parent directory and src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, 'src')
sys.path.insert(0, parent_dir)
sys.path.insert(0, src_dir)

from src.evaluate import ModelEvaluator
from src.data_loader import CervicalDataLoader
from src.visualization import ResultVisualizer, GradCAM
from src.utils import setup_logging, load_config, set_seeds
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)


def main(config_path=None, model_path=None):
    """
    Main evaluation function
    
    Args:
        config_path (str): Path to configuration file
        model_path (str): Path to model file (optional)
    """
    # Setup logging
    setup_logging()
    
    logger.info("="*60)
    logger.info("CERVICAL CYTOLOGY CLASSIFICATION - EVALUATION")
    logger.info("="*60)
    
    # Default config path
    if config_path is None:
        config_path = os.path.join(parent_dir, "config", "config.yaml")
    
    # Load configuration
    logger.info(f"Loading configuration from: {config_path}")
    config = load_config(config_path)
    
    # Set seeds
    set_seeds(config['dataset']['seed'])
    
    # Determine model path
    if model_path is None:
        model_path = os.path.join(parent_dir, config['paths']['models_dir'], 'best_model.h5')
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found at: {model_path}")
        logger.error("Please train the model first using: python scripts/train_model.py")
        sys.exit(1)
    
    # Load model
    logger.info(f"\nLoading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    logger.info("Model loaded successfully")
    
    # Load test data
    logger.info("\nLoading test dataset...")
    data_loader = CervicalDataLoader(config)
    data = data_loader.prepare_dataset()
    
    logger.info(f"Test samples: {len(data['X_test'])}")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model, config)
    
    # Compute all metrics
    logger.info("\nComputing evaluation metrics...")
    results = evaluator.compute_all_metrics(data['X_test'], data['y_test'])
    
    # Save results
    results_dir = os.path.join(parent_dir, config['paths']['reports_dir'])
    evaluator.save_results(results, results_dir)
    
    # Print results summary
    logger.info("\n" + "="*60)
    logger.info("EVALUATION RESULTS")
    logger.info("="*60)
    logger.info(f"\nOverall Metrics:")
    logger.info(f"  Test Accuracy:  {results.get('test_accuracy', 0.0):.4f}")
    logger.info(f"  Test Loss:      {results.get('test_loss', 0.0):.4f}")
    if 'cohens_kappa' in results:
        logger.info(f"  Cohen's Kappa:  {results['cohens_kappa']:.4f}")
    if 'roc_auc' in results and 'macro' in results['roc_auc']:
        logger.info(f"  ROC-AUC (macro): {results['roc_auc']['macro']:.4f}")
    
    # Per-class metrics
    logger.info(f"\nPer-Class Metrics:")
    logger.info(f"{'Class':<30} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    logger.info("-"*66)
    
    for class_name in config['dataset']['classes']:
        if class_name in results['per_class_metrics']:
            metrics = results['per_class_metrics'][class_name]
            logger.info(
                f"{class_name:<30} "
                f"{metrics['precision']:<12.4f} "
                f"{metrics['recall']:<12.4f} "
                f"{metrics['f1_score']:<12.4f}"
            )
    
    logger.info("="*60)
    
    # Create visualizations
    logger.info("\nGenerating visualizations...")
    visualizer = ResultVisualizer(config, config['dataset']['classes'])
    
    # Plot confusion matrix
    if 'confusion_matrix' in results:
        logger.info("Creating confusion matrix plots...")
        visualizer.plot_confusion_matrix(results['confusion_matrix'], normalize=False, save=True)
        visualizer.plot_confusion_matrix(results['confusion_matrix'], normalize=True, save=True)
    
    # Plot ROC curves
    if 'prediction_probabilities' in results:
        logger.info("Creating ROC curves...")
        visualizer.plot_roc_curves(
            data['y_test'],
            results['prediction_probabilities'],
            save=True
        )
    
    # Plot sample predictions
    logger.info("Creating sample predictions plot...")
    num_samples = min(10, len(data['X_test']))
    visualizer.plot_sample_predictions(
        data['X_test'],
        results['true_labels'],
        results['predictions'],
        results['prediction_probabilities'],
        num_samples=num_samples,
        save=True
    )
    
    # Generate Grad-CAM visualizations
    if config['evaluation']['generate_gradcam']:
        logger.info("\nGenerating Grad-CAM visualizations...")
        try:
            gradcam = GradCAM(model)
            
            # Select random samples
            num_gradcam_samples = min(
                config['evaluation'].get('num_samples_visualize', 5),
                len(data['X_test'])
            )
            
            indices = np.random.choice(len(data['X_test']), num_gradcam_samples, replace=False)
            
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(num_gradcam_samples, 3, figsize=(15, num_gradcam_samples * 5))
            
            if num_gradcam_samples == 1:
                axes = axes.reshape(1, -1)
            
            for i, idx in enumerate(indices):
                image = data['X_test'][idx]
                true_class = results['true_labels'][idx]
                pred_class = results['predictions'][idx]
                
                # Compute heatmap
                heatmap = gradcam.compute_heatmap(image, pred_class)
                overlayed = gradcam.overlay_heatmap(image, heatmap)
                
                # Plot
                axes[i, 0].imshow(image)
                axes[i, 0].set_title(f'Original\nTrue: {config["dataset"]["classes"][true_class]}')
                axes[i, 0].axis('off')
                
                axes[i, 1].imshow(heatmap, cmap='jet')
                axes[i, 1].set_title('Grad-CAM Heatmap')
                axes[i, 1].axis('off')
                
                axes[i, 2].imshow(overlayed)
                axes[i, 2].set_title(f'Overlay\nPred: {config["dataset"]["classes"][pred_class]}')
                axes[i, 2].axis('off')
            
            plt.tight_layout()
            gradcam_path = os.path.join(parent_dir, config['paths']['plots_dir'], 'gradcam_visualizations.png')
            plt.savefig(gradcam_path, dpi=300, bbox_inches='tight')
            logger.info(f"Grad-CAM visualizations saved to: {gradcam_path}")
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not generate Grad-CAM visualizations: {e}")
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("EVALUATION COMPLETED SUCCESSFULLY")
    logger.info("="*60)
    logger.info(f"\nResults saved to: {results_dir}")
    logger.info(f"Plots saved to: {os.path.join(parent_dir, config['paths']['plots_dir'])}")
    logger.info("\nGenerated files:")
    logger.info("  - evaluation_results.json")
    logger.info("  - classification_report.txt")
    logger.info("  - confusion_matrix.png")
    logger.info("  - confusion_matrix_normalized.png")
    logger.info("  - roc_curves.png")
    logger.info("  - sample_predictions.png")
    if config['evaluation']['generate_gradcam']:
        logger.info("  - gradcam_visualizations.png")
    logger.info("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate cervical cytology classification model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model file (default: models/saved_models/best_model.h5)"
    )
    
    args = parser.parse_args()
    
    main(config_path=args.config, model_path=args.model)