import os

import tensorflow as tf

from classification import EfficientNet, MobileNet, ResNet152V2


def get_classification_model_callbacks(
        model_name: str,
        early_stop_patience: int,
        reduce_lr_patience: int,
        checkpoint_save_frequency: int) -> [tf.keras.callbacks.Callback]:
    model_dir_path = os.path.join('out', 'training-callback-results', model_name)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir_path, 'model_checkpoints'),
            monitor='val_accuracy',
            mode='max'
        ),
        tf.keras.callbacks.BackupAndRestore(
            backup_dir=os.path.join(model_dir_path, 'backup'),
            save_freq=checkpoint_save_frequency
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            min_delta=0.001,
            patience=early_stop_patience,
            start_from_epoch=10
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(model_dir_path, 'tensorboard-logs'),
            histogram_freq=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=reduce_lr_patience,
            cooldown=2,
            min_lr=0.001
        )
    ]


def get_segmentation_model_callbacks(
        model_name: str,
        early_stop_patience: int,
        reduce_lr_patience: int,
        checkpoint_save_frequency: int) -> [tf.keras.callbacks.Callback]:
    model_dir_path = os.path.join('out', 'training-callback-results', model_name)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir_path, 'checkpoint'),
            monitor='val_io_u',
            mode='max',
        ),
        tf.keras.callbacks.BackupAndRestore(
            backup_dir=os.path.join(model_dir_path, 'backup'),
            save_freq=checkpoint_save_frequency
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='io_u',
            min_delta=0.0001,
            patience=early_stop_patience,
            start_from_epoch=10
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(model_dir_path, 'tensorboard-logs'),
            histogram_freq=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_io_u',
            factor=0.5,
            patience=reduce_lr_patience,
            cooldown=2,
            min_lr=0.001
        )
    ]
