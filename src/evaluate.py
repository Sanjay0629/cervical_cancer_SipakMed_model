"""
Model evaluation module for cervical cytology classification
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report, confusion_matrix,
    cohen_kappa_score, roc_auc_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import logging
import json

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Model evaluation and metrics computation
    """
    
    def __init__(self, model, config):
        """
        Initialize evaluator
        
        Args:
            model: Trained Keras model
            config (dict): Configuration dictionary
        """
        self.model = model
        self.config = config
        self.class_names = config['dataset']['classes']
        self.num_classes = len(self.class_names)
        
    def evaluate_on_test_set(self, X_test, y_test):
        """
        Evaluate model on test set
        
        Args:
            X_test (np.array): Test images
            y_test (np.array): Test labels (one-hot)
            
        Returns:
            dict: Evaluation metrics
        """
        logger.info("Evaluating model on test set...")
        
        # Get predictions
        y_pred_probs = self.model.predict(
            X_test,
            batch_size=self.config['evaluation']['batch_size'],
            verbose=1
        )
        y_pred_classes = np.argmax(y_pred_probs, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Compute metrics
        results = {
            'test_loss': 0.0,  # Will be computed
            'test_accuracy': 0.0,
            'predictions': y_pred_classes,
            'prediction_probabilities': y_pred_probs,
            'true_labels': y_true_classes
        }
        
        # Evaluate using model.evaluate
        test_metrics = self.model.evaluate(
            X_test, y_test,
            batch_size=self.config['evaluation']['batch_size'],
            verbose=1
        )
        
        # Extract metrics
        metric_names = self.model.metrics_names
        for name, value in zip(metric_names, test_metrics):
            results[f'test_{name}'] = float(value)
        
        logger.info(f"Test Accuracy: {results.get('test_accuracy', 0.0):.4f}")
        
        return results
    
    def compute_confusion_matrix(self, y_true, y_pred):
        """
        Compute confusion matrix
        
        Args:
            y_true (np.array): True labels
            y_pred (np.array): Predicted labels
            
        Returns:
            np.array: Confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        logger.info("Confusion matrix computed")
        return cm
    
    def compute_classification_report(self, y_true, y_pred):
        """
        Compute classification report
        
        Args:
            y_true (np.array): True labels
            y_pred (np.array): Predicted labels
            
        Returns:
            dict: Classification report
        """
        report = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )
        logger.info("Classification report computed")
        return report
    
    def compute_per_class_metrics(self, y_true, y_pred):
        """
        Compute per-class metrics
        
        Args:
            y_true (np.array): True labels
            y_pred (np.array): Predicted labels
            
        Returns:
            dict: Per-class metrics
        """
        metrics = {}
        
        for i, class_name in enumerate(self.class_names):
            # True positives, false positives, false negatives
            tp = np.sum((y_true == i) & (y_pred == i))
            fp = np.sum((y_true != i) & (y_pred == i))
            fn = np.sum((y_true == i) & (y_pred != i))
            tn = np.sum((y_true != i) & (y_pred != i))
            
            # Compute metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            metrics[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'specificity': specificity,
                'support': int(np.sum(y_true == i))
            }
        
        return metrics
    
    def compute_cohens_kappa(self, y_true, y_pred):
        """
        Compute Cohen's Kappa score
        
        Args:
            y_true (np.array): True labels
            y_pred (np.array): Predicted labels
            
        Returns:
            float: Cohen's Kappa score
        """
        kappa = cohen_kappa_score(y_true, y_pred)
        logger.info(f"Cohen's Kappa: {kappa:.4f}")
        return kappa
    
    def compute_roc_auc(self, y_true_onehot, y_pred_probs):
        """
        Compute ROC-AUC scores (one-vs-rest)
        
        Args:
            y_true_onehot (np.array): True labels (one-hot)
            y_pred_probs (np.array): Prediction probabilities
            
        Returns:
            dict: ROC-AUC scores
        """
        roc_auc_scores = {}
        
        # Overall ROC-AUC (macro average)
        try:
            roc_auc_macro = roc_auc_score(
                y_true_onehot,
                y_pred_probs,
                average='macro',
                multi_class='ovr'
            )
            roc_auc_scores['macro'] = roc_auc_macro
        except Exception as e:
            logger.warning(f"Could not compute macro ROC-AUC: {e}")
            roc_auc_scores['macro'] = 0.0
        
        # Per-class ROC-AUC
        for i, class_name in enumerate(self.class_names):
            try:
                roc_auc_scores[class_name] = roc_auc_score(
                    y_true_onehot[:, i],
                    y_pred_probs[:, i]
                )
            except Exception as e:
                logger.warning(f"Could not compute ROC-AUC for {class_name}: {e}")
                roc_auc_scores[class_name] = 0.0
        
        logger.info(f"ROC-AUC (macro): {roc_auc_scores.get('macro', 0.0):.4f}")
        
        return roc_auc_scores
    
    def compute_all_metrics(self, X_test, y_test):
        """
        Compute all evaluation metrics
        
        Args:
            X_test (np.array): Test images
            y_test (np.array): Test labels (one-hot)
            
        Returns:
            dict: All metrics
        """
        # Evaluate on test set
        results = self.evaluate_on_test_set(X_test, y_test)
        
        y_true = results['true_labels']
        y_pred = results['predictions']
        y_pred_probs = results['prediction_probabilities']
        
        # Confusion matrix
        if self.config['evaluation']['compute_confusion_matrix']:
            results['confusion_matrix'] = self.compute_confusion_matrix(y_true, y_pred)
        
        # Classification report
        if self.config['evaluation']['compute_classification_report']:
            results['classification_report'] = self.compute_classification_report(y_true, y_pred)
        
        # Per-class metrics
        results['per_class_metrics'] = self.compute_per_class_metrics(y_true, y_pred)
        
        # Cohen's Kappa
        if self.config['evaluation']['compute_cohens_kappa']:
            results['cohens_kappa'] = self.compute_cohens_kappa(y_true, y_pred)
        
        # ROC-AUC
        if self.config['evaluation']['compute_roc_auc']:
            results['roc_auc'] = self.compute_roc_auc(y_test, y_pred_probs)
        
        return results
    
    def save_results(self, results, save_dir):
        """
        Save evaluation results to file
        
        Args:
            results (dict): Evaluation results
            save_dir (str): Directory to save results
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Prepare results for JSON serialization
        json_results = {}
        
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            elif isinstance(value, (np.int64, np.int32)):
                json_results[key] = int(value)
            elif isinstance(value, (np.float64, np.float32)):
                json_results[key] = float(value)
            else:
                json_results[key] = value
        
        # Save to JSON
        json_path = os.path.join(save_dir, 'evaluation_results.json')
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=4)
        
        logger.info(f"Results saved to: {json_path}")
        
        # Save classification report separately
        if 'classification_report' in results:
            report_path = os.path.join(save_dir, 'classification_report.txt')
            with open(report_path, 'w') as f:
                f.write("CLASSIFICATION REPORT\n")
                f.write("="*60 + "\n\n")
                
                report = results['classification_report']
                
                # Header
                f.write(f"{'Class':<30} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
                f.write("-"*76 + "\n")
                
                # Per-class metrics
                for class_name in self.class_names:
                    if class_name in report:
                        metrics = report[class_name]
                        f.write(f"{class_name:<30} ")
                        f.write(f"{metrics['precision']:<12.4f} ")
                        f.write(f"{metrics['recall']:<12.4f} ")
                        f.write(f"{metrics['f1-score']:<12.4f} ")
                        f.write(f"{metrics['support']:<10.0f}\n")
                
                # Overall metrics
                f.write("-"*76 + "\n")
                if 'accuracy' in report:
                    f.write(f"{'Accuracy':<30} {'':<12} {'':<12} {report['accuracy']:<12.4f} {report.get('support', 0):<10.0f}\n")
                if 'macro avg' in report:
                    avg = report['macro avg']
                    f.write(f"{'Macro Average':<30} {avg['precision']:<12.4f} {avg['recall']:<12.4f} {avg['f1-score']:<12.4f} {avg['support']:<10.0f}\n")
                if 'weighted avg' in report:
                    avg = report['weighted avg']
                    f.write(f"{'Weighted Average':<30} {avg['precision']:<12.4f} {avg['recall']:<12.4f} {avg['f1-score']:<12.4f} {avg['support']:<10.0f}\n")
                
                # Cohen's Kappa
                if 'cohens_kappa' in results:
                    f.write("\n" + "-"*76 + "\n")
                    f.write(f"Cohen's Kappa: {results['cohens_kappa']:.4f}\n")
            
            logger.info(f"Classification report saved to: {report_path}")
        
        return json_path


def main():
    """
    Main evaluation function
    """
    from utils import setup_logging, load_config, set_seeds
    from data_loader import CervicalDataLoader
    
    # Setup logging
    logger_instance = setup_logging()
    
    # Load configuration
    config = load_config("config/config.yaml")
    
    # Set seeds
    set_seeds(config['dataset']['seed'])
    
    # Load model
    model_path = os.path.join(config['paths']['models_dir'], 'best_model.h5')
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found at: {model_path}")
        logger.error("Please train the model first using train.py")
        return
    
    logger.info(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # Load data
    logger.info("Loading test data...")
    data_loader = CervicalDataLoader(config)
    data = data_loader.prepare_dataset()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model, config)
    
    # Compute all metrics
    logger.info("Computing evaluation metrics...")
    results = evaluator.compute_all_metrics(data['X_test'], data['y_test'])
    
    # Save results
    results_dir = config['paths']['reports_dir']
    evaluator.save_results(results, results_dir)
    
    # Print summary
    logger.info("="*60)
    logger.info("EVALUATION COMPLETED")
    logger.info("="*60)
    logger.info(f"Test Accuracy: {results.get('test_accuracy', 0.0):.4f}")
    logger.info(f"Cohen's Kappa: {results.get('cohens_kappa', 0.0):.4f}")
    if 'roc_auc' in results:
        logger.info(f"ROC-AUC (macro): {results['roc_auc'].get('macro', 0.0):.4f}")
    logger.info("="*60)


if __name__ == "__main__":
    main()