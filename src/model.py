"""
Model architecture module for cervical cytology classification
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import logging

logger = logging.getLogger(__name__)


class CervicalCNNModel:
    """
    Cervical cytology CNN model builder
    """
    
    def __init__(self, config):
        """
        Initialize model builder
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        self.input_shape = tuple(config['model']['input_shape'])
        self.num_classes = config['model']['num_classes']
        self.base_model_name = config['model']['base_model']
        
    def get_base_model(self):
        """
        Get pretrained base model
        
        Returns:
            keras.Model: Base model
        """
        weights = self.config['model']['pretrained_weights']
        
        base_models = {
            'ResNet50': keras.applications.ResNet50,
            'MobileNetV2': keras.applications.MobileNetV2,
            'EfficientNetB0': keras.applications.EfficientNetB0,
            'DenseNet121': keras.applications.DenseNet121,
            'InceptionV3': keras.applications.InceptionV3,
            'VGG16': keras.applications.VGG16,
        }
        
        if self.base_model_name not in base_models:
            raise ValueError(f"Base model {self.base_model_name} not supported")
        
        base_model_class = base_models[self.base_model_name]
        
        base_model = base_model_class(
            include_top=False,
            weights=weights,
            input_shape=self.input_shape,
            pooling=None
        )
        
        logger.info(f"Loaded base model: {self.base_model_name}")
        logger.info(f"Total parameters: {base_model.count_params():,}")
        
        return base_model
    
    def build_classification_head(self, base_model):
        """
        Build custom classification head
        
        Args:
            base_model: Base model
            
        Returns:
            keras.Model: Complete model
        """
        # Get configuration
        dense_units = self.config['model']['dense_units']
        dropout_rates = self.config['model']['dropout_rates']
        use_batch_norm = self.config['model']['use_batch_norm']
        
        # Build model
        inputs = keras.Input(shape=self.input_shape)
        
        # Base model
        x = base_model(inputs, training=False)
        
        # Global Average Pooling
        x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
        
        # Dense layers with dropout
        for i, (units, dropout_rate) in enumerate(zip(dense_units, dropout_rates)):
            x = layers.Dense(units, activation='relu', name=f'dense_{i+1}')(x)
            
            if use_batch_norm:
                x = layers.BatchNormalization(name=f'bn_{i+1}')(x)
            
            x = layers.Dropout(dropout_rate, name=f'dropout_{i+1}')(x)
        
        # Output layer
        outputs = layers.Dense(
            self.num_classes,
            activation='softmax',
            name='predictions'
        )(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name='cervical_classifier')
        
        logger.info("Classification head built successfully")
        
        return model
    
    def build_model(self, freeze_base=True):
        """
        Build complete model with base + classification head
        
        Args:
            freeze_base (bool): Whether to freeze base model layers
            
        Returns:
            keras.Model: Complete model
        """
        # Get base model
        base_model = self.get_base_model()
        
        # Freeze base model if specified
        if freeze_base:
            base_model.trainable = False
            logger.info("Base model frozen")
        else:
            base_model.trainable = True
            logger.info("Base model unfrozen")
        
        # Build complete model
        model = self.build_classification_head(base_model)
        
        # Print model summary
        logger.info(f"Total parameters: {model.count_params():,}")
        logger.info(f"Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
        
        return model
    
    def unfreeze_layers(self, model, num_layers):
        """
        Unfreeze last N layers of the base model for fine-tuning
        
        Args:
            model: Keras model
            num_layers (int): Number of layers to unfreeze from the end
        """
        # Get base model (first layer after input)
        base_model = model.layers[1] if len(model.layers) > 1 else model
        
        # Make base model trainable
        base_model.trainable = True
        
        # Freeze all layers except the last num_layers
        total_layers = len(base_model.layers)
        freeze_until = max(0, total_layers - num_layers)
        
        for i, layer in enumerate(base_model.layers):
            if i < freeze_until:
                layer.trainable = False
            else:
                layer.trainable = True
        
        logger.info(f"Unfroze last {num_layers} layers of base model")
        logger.info(f"Total layers: {total_layers}, Frozen: {freeze_until}, Trainable: {num_layers}")


def compile_model(model, config, stage='stage1'):
    """
    Compile model with optimizer, loss, and metrics
    
    Args:
        model: Keras model
        config (dict): Configuration dictionary
        stage (str): Training stage ('stage1' or 'stage2')
        
    Returns:
        keras.Model: Compiled model
    """
    stage_config = config['training'][stage]
    learning_rate = stage_config['learning_rate']
    
    # Optimizer
    optimizer_name = config['training']['optimizer'].lower()
    
    if optimizer_name == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=0.9,
            nesterov=True
        )
    elif optimizer_name == 'adamw':
        optimizer = keras.optimizers.AdamW(learning_rate=learning_rate)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported")
    
    # Loss
    loss = config['training']['loss']
    
    # Metrics
    metrics = [
        'accuracy',
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
    ]
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    logger.info(f"Model compiled for {stage}")
    logger.info(f"Optimizer: {optimizer_name}, LR: {learning_rate}")
    
    return model


def create_callbacks(config, stage='stage1'):
    """
    Create training callbacks
    
    Args:
        config (dict): Configuration dictionary
        stage (str): Training stage
        
    Returns:
        list: List of callbacks
    """
    callbacks = []
    
    # Early Stopping
    if config['callbacks']['early_stopping']:
        early_stop_config = config['callbacks']['early_stopping']
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor=early_stop_config['monitor'],
                patience=early_stop_config['patience'],
                restore_best_weights=early_stop_config['restore_best_weights'],
                min_delta=early_stop_config.get('min_delta', 0.001),
                verbose=1
            )
        )
        logger.info("Early stopping callback added")
    
    # Reduce Learning Rate on Plateau
    if config['callbacks']['reduce_lr']:
        reduce_lr_config = config['callbacks']['reduce_lr']
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor=reduce_lr_config['monitor'],
                factor=reduce_lr_config['factor'],
                patience=reduce_lr_config['patience'],
                min_lr=reduce_lr_config['min_lr'],
                verbose=reduce_lr_config.get('verbose', 1)
            )
        )
        logger.info("ReduceLROnPlateau callback added")
    
    # Model Checkpoint
    if config['callbacks']['model_checkpoint']:
        checkpoint_config = config['callbacks']['model_checkpoint']
        checkpoint_path = f"{config['paths']['checkpoints_dir']}/best_model_{stage}.h5"
        
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor=checkpoint_config['monitor'],
                save_best_only=checkpoint_config['save_best_only'],
                mode=checkpoint_config['mode'],
                save_weights_only=checkpoint_config.get('save_weights_only', False),
                verbose=1
            )
        )
        logger.info(f"Model checkpoint callback added: {checkpoint_path}")
    
    # TensorBoard
    if config['callbacks']['tensorboard']:
        tensorboard_config = config['callbacks']['tensorboard']
        log_dir = f"{tensorboard_config['log_dir']}/{stage}"
        
        callbacks.append(
            keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=tensorboard_config.get('histogram_freq', 1),
                write_graph=True,
                write_images=False,
                update_freq='epoch'
            )
        )
        logger.info(f"TensorBoard callback added: {log_dir}")
    
    # CSV Logger
    csv_path = f"{config['paths']['logs_dir']}/training_log_{stage}.csv"
    callbacks.append(
        keras.callbacks.CSVLogger(csv_path, separator=',', append=False)
    )
    logger.info(f"CSV logger added: {csv_path}")
    
    return callbacks


def get_model_summary(model, print_summary=True):
    """
    Get model summary information
    
    Args:
        model: Keras model
        print_summary (bool): Whether to print summary
        
    Returns:
        dict: Model summary information
    """
    total_params = model.count_params()
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    summary_info = {
        'total_parameters': int(total_params),
        'trainable_parameters': int(trainable_params),
        'non_trainable_parameters': int(non_trainable_params),
        'num_layers': len(model.layers),
        'input_shape': model.input_shape,
        'output_shape': model.output_shape
    }
    
    if print_summary:
        print("\n" + "="*60)
        print("MODEL SUMMARY")
        print("="*60)
        print(f"Total Parameters:         {total_params:,}")
        print(f"Trainable Parameters:     {trainable_params:,}")
        print(f"Non-trainable Parameters: {non_trainable_params:,}")
        print(f"Number of Layers:         {len(model.layers)}")
        print(f"Input Shape:              {model.input_shape}")
        print(f"Output Shape:             {model.output_shape}")
        print("="*60 + "\n")
    
    return summary_info


if __name__ == "__main__":
    # Test model building
    import yaml
    from utils import setup_logging, set_seeds
    
    setup_logging()
    
    # Load config
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    set_seeds(config['dataset']['seed'])
    
    # Build model
    model_builder = CervicalCNNModel(config)
    model = model_builder.build_model(freeze_base=True)
    
    # Compile model
    model = compile_model(model, config, stage='stage1')
    
    # Get summary
    summary_info = get_model_summary(model)
    
    # Print detailed summary
    model.summary()
    
    print("\nModel built and compiled successfully!")