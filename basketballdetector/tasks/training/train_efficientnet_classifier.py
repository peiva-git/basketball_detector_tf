import pathlib

from basketballdetector.data import ClassificationSequenceBuilder
from basketballdetector.models.callbacks import get_classification_model_callbacks
from basketballdetector.models.classification import EfficientNet

import tensorflow as tf

from .saving import save_models


def train_and_save_model(dataset_path: str,
                         batch_size: int,
                         image_width: int,
                         image_height: int,
                         validation_percentage: float = 0.2,
                         learning_rate: float = 0.01,
                         epochs: int = 100,
                         early_stop_patience: int = 10,
                         reduce_lr_patience: int = 5,
                         checkpoint_save_frequency: int = 10000):
    """
    Train an EfficientNet binary classifier. After the training is completed, the model is saved using the
    `basketballdetector.tasks.training.saving.save_models` function.
    :param dataset_path: The path to the dataset root. The dataset structure must comply to the specifications
    provided by the `basketballdetector.data.sequence_builders.ClassificationSequenceBuilder` class
    :param batch_size: Batch size used by the dataset Sequence
    :param image_width: Dataset image width
    :param image_height: Dataset image height
    :param validation_percentage: Percentage of images to be used as validation.
    The remaining images will be used for training
    :param learning_rate: Learning rate of the optimizer function
    :param epochs: Number of epochs o train the model
    :param early_stop_patience: Number of epochs to wait before stopping the training in case of no improvement.
    For more details, take a look here https://keras.io/api/callbacks/early_stopping/
    :param reduce_lr_patience: Number of epochs to wait before reducing the learning rate in case if no improvement.
    For more details, take a look here https://keras.io/api/callbacks/reduce_lr_on_plateau/
    :param checkpoint_save_frequency: A model checkpoint will be saved in out/training-callback-results after
    the specified number of iterations. You can also specify 'epoch' instead.
    See https://keras.io/api/callbacks/model_checkpoint/
    :return: None
    """
    dataset_dir = pathlib.Path(dataset_path)
    builder = ClassificationSequenceBuilder(
        str(dataset_dir),
        batch_size,
        image_width,
        image_height,
        validation_percentage=validation_percentage
    )
    train_sequence, val_sequence = builder.training_sequence, builder.validation_sequence

    classifier = EfficientNet(number_of_classes=2)
    classifier.model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.AUC(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()
        ]
    )

    classifier.model.fit(
        train_sequence,
        validation_data=val_sequence,
        epochs=epochs,
        callbacks=get_classification_model_callbacks(
            classifier.model_name,
            early_stop_patience=early_stop_patience,
            reduce_lr_patience=reduce_lr_patience,
            checkpoint_save_frequency=checkpoint_save_frequency
        )
    )

    save_models(classifier)


if __name__ == '__main__':
    train_and_save_model(
        '/home/ubuntu/dataset_classification/pallacanestro_trieste',
        batch_size=8,
        image_width=112,
        image_height=112
    )
