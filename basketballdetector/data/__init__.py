import tensorflow as tf

from .dataset_builders import ClassificationDatasetBuilder, SegmentationDatasetBuilder
from .sequence_builders import ClassificationSequenceBuilder, PatchesSequence


def decode_image(image_data, image_width: int = 50, image_height: int = 50, channels: int = 3):
    image = tf.io.decode_png(image_data, channels=channels)
    return tf.image.resize(image, [image_height, image_width])
