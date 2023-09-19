import pathlib

from basketballdetector.data import ClassificationSequenceBuilder
from basketballdetector.models.callbacks import get_classification_model_callbacks
from basketballdetector.models.classification import EfficientNet

import tensorflow as tf

from .saving import save_models


def train_and_save_model(dataset_path: str,
                         batch_size: int,
                         number_of_classes: int,
                         validation_percentage: float = 0.2,
                         learning_rate: float = 0.01,
                         epochs: int = 100,
                         early_stop_patience: int = 10,
                         reduce_lr_patience: int = 5,
                         checkpoint_save_frequency: int = 10000):
    dataset_dir = pathlib.Path(dataset_path)
    builder = ClassificationSequenceBuilder(
        str(dataset_dir),
        batch_size,
        validation_percentage=validation_percentage
    )
    train_sequence, val_sequence = builder.training_sequence, builder.validation_sequence

    classifier = EfficientNet(number_of_classes=number_of_classes)
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
        number_of_classes=2
    )
