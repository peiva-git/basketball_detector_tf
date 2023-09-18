from basketballdetector.data import ClassificationSequenceBuilder
from basketballdetector.models.callbacks import get_classification_model_callbacks
from basketballdetector.models.classification import EfficientNet

import tensorflow as tf

from basketballdetector.tasks.training.saving import save_models


def train_and_save_model():
    builder = ClassificationSequenceBuilder('/home/ubuntu/classification_dataset/pallacanestro_trieste/', 8)
    train_sequence, val_sequence = builder.training_sequence, builder.validation_sequence

    classifier = EfficientNet(number_of_classes=2)
    classifier.model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
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
        epochs=100,
        callbacks=get_classification_model_callbacks(
            classifier.model_name,
            early_stop_patience=10,
            reduce_lr_patience=5,
            checkpoint_save_frequency=10000
        )
    )

    save_models(classifier)


if __name__ == '__main__':
    train_and_save_model()
