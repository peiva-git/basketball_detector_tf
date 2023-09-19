import tensorflow as tf

from basketballdetector.data import SegmentationDatasetBuilder
from basketballdetector.models import UNet
from basketballdetector.models.callbacks import get_segmentation_model_callbacks
from .saving import save_models


def train_and_save_model():
    builder = SegmentationDatasetBuilder('/home/ubuntu/segmentation_dataset/pallacanestro_trieste')
    builder.configure_datasets_for_performance(shuffle_buffer_size=50, input_batch_size=4)
    train_dataset, validation_dataset = builder.train_dataset, builder.validation_dataset

    segmentor = UNet(input_shape=(512, 1024, 3), number_of_classes=2)
    segmentor.model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.SGD(momentum=0.9, learning_rate=0.01),
        metrics=['accuracy', tf.keras.metrics.IoU(num_classes=2, target_class_ids=[1])]
    )
    segmentor.model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=100,
        callbacks=get_segmentation_model_callbacks(
            segmentor.model_name,
            early_stop_patience=12,
            reduce_lr_patience=3,
            checkpoint_save_frequency='epoch'
        )
    )
    save_models(segmentor)


if __name__ == '__main__':
    train_and_save_model()
