import tensorflow as tf


def decode_image(image_data: tf.Tensor, image_width: int = 50, image_height: int = 50, channels: int = 3) -> tf.Tensor:
    image = tf.io.decode_png(image_data, channels=channels)
    return tf.image.resize(image, [image_height, image_width])
