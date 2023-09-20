"""
This module contains utility functions to obtain useful callbacks to be used during model training.
"""

import os

import tensorflow as tf


def get_classification_model_callbacks(
        model_name: str,
        early_stop_patience: int,
        reduce_lr_patience: int,
        checkpoint_save_frequency) -> [tf.keras.callbacks.Callback]:
    """
    This function returns a list of the following callbacks:
    1. ModelCheckpoint https://keras.io/api/callbacks/model_checkpoint/
    2. BackupAndRestore https://keras.io/api/callbacks/backup_and_restore/
    3. EarlyStopping https://keras.io/api/callbacks/early_stopping/
    4. TensorBoard https://keras.io/api/callbacks/tensorboard/
    5. ReduceLROnPlateau https://keras.io/api/callbacks/reduce_lr_on_plateau/
    :param model_name: The model name. Used to know where to save the callbacks' outputs
    :param early_stop_patience: See https://keras.io/api/callbacks/early_stopping/
    :param reduce_lr_patience: See https://keras.io/api/callbacks/reduce_lr_on_plateau/
    :param checkpoint_save_frequency: See https://keras.io/api/callbacks/model_checkpoint/
    :return: A list of `Callback`s
    """
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
        checkpoint_save_frequency) -> [tf.keras.callbacks.Callback]:
    """
    This function returns a list of the following callbacks:
    1. ModelCheckpoint https://keras.io/api/callbacks/model_checkpoint/
    2. BackupAndRestore https://keras.io/api/callbacks/backup_and_restore/
    3. EarlyStopping https://keras.io/api/callbacks/early_stopping/
    4. TensorBoard https://keras.io/api/callbacks/tensorboard/
    5. ReduceLROnPlateau https://keras.io/api/callbacks/reduce_lr_on_plateau/
    :param model_name: The model name. Used to know where to save the callbacks' outputs
    :param early_stop_patience: See https://keras.io/api/callbacks/early_stopping/
    :param reduce_lr_patience: See https://keras.io/api/callbacks/reduce_lr_on_plateau/
    :param checkpoint_save_frequency: See https://keras.io/api/callbacks/model_checkpoint/
    :return: A list of `Callback`s
    """
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
