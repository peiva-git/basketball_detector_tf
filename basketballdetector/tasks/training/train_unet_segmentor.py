"""
This module is used to train an UNet binary segmentor.
It can be executed as a script to run with default parameters.
"""

import pathlib

import tensorflow as tf

from basketballdetector.data import SegmentationDatasetBuilder
from basketballdetector.models import UNet
from basketballdetector.models.callbacks import get_segmentation_model_callbacks
from .saving import save_models


def train_and_save_model(dataset_path: str,
                         batch_size: int,
                         shuffle_buffer_size: int,
                         image_width: int,
                         image_height: int,
                         validation_percentage: float = 0.2,
                         learning_rate: float = 0.01,
                         momentum: float = 0.9,
                         epochs: int = 100,
                         early_stop_patience: int = 12,
                         reduce_lr_patience: int = 3,
                         checkpoint_save_frequency='epoch'
                         ):
    """
    Train an UNet binary segmentor with an SGD optimizer. After the training is completed, the model is saved using the
    `basketballdetector.tasks.training.saving.save_models` function.
    :param dataset_path: The path to the dataset root. The dataset structure must comply to the specifications
    provided by the `basketballdetector.data.dataset_builders.SegmentationDatasetBuilder` class
    :param batch_size: Batch size used by the Dataset
    :param shuffle_buffer_size: Size of the shuffle buffer.
    See [here](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shuffle)
    :param image_width: Dataset image width
    :param image_height: Dataset image height
    :param validation_percentage: Percentage of images to be used as validation.
    The remaining images will be used for training
    :param learning_rate: Learning rate of the optimizer function
    :param momentum: Momentum of the optimizer function
    :param epochs: Number of epochs to train the model
    :param early_stop_patience: Number of epochs to wait before stopping the training in case of no improvement.
    For more details, take a look [here](https://keras.io/api/callbacks/early_stopping/)
    :param reduce_lr_patience: Number of epochs to wait before reducing the learning rate in case if no improvement.
    For more details, take a look [here](https://keras.io/api/callbacks/reduce_lr_on_plateau/)
    :param checkpoint_save_frequency: A model checkpoint will be saved in `out/training-callback-results` after
    the specified number of iterations. You can also specify 'epoch' instead.
    See [here](https://keras.io/api/callbacks/model_checkpoint/)
    :return: None
    """
    dataset_dir = pathlib.Path(dataset_path)
    builder = SegmentationDatasetBuilder(
        str(dataset_dir),
        image_width=image_width,
        image_height=image_height,
        validation_percentage=validation_percentage
    )
    builder.configure_datasets_for_performance(shuffle_buffer_size=shuffle_buffer_size, input_batch_size=batch_size)
    train_dataset, validation_dataset = builder.train_dataset, builder.validation_dataset

    segmentor = UNet(input_shape=(image_height, image_width, 3), number_of_classes=2)
    segmentor.model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.SGD(momentum=momentum, learning_rate=learning_rate),
        metrics=['accuracy', tf.keras.metrics.IoU(num_classes=2, target_class_ids=[1])]
    )
    segmentor.model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=get_segmentation_model_callbacks(
            segmentor.model_name,
            early_stop_patience=early_stop_patience,
            reduce_lr_patience=reduce_lr_patience,
            checkpoint_save_frequency=checkpoint_save_frequency
        )
    )
    save_models(segmentor)


if __name__ == '__main__':
    train_and_save_model(
        '/home/ubuntu/segmentation_dataset/pallacanestro_trieste',
        batch_size=4,
        shuffle_buffer_size=50,
        image_width=1024,
        image_height=512
    )
