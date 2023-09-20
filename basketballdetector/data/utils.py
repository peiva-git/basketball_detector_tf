import tensorflow as tf


def decode_image(image_data: tf.Tensor, image_width: int = 50, image_height: int = 50, channels: int = 3) -> tf.Tensor:
    """
    This function decodes an image from raw input data in the form of a Tensor.
    It also resizes the image to the specified width and height-
    :param image_data: Raw image data
    :param image_width: Target image width
    :param image_height: Target image height
    :param channels: Target image color channels
    :return: The decoded image
    """
    image = tf.io.decode_png(image_data, channels=channels)
    return tf.image.resize(image, [image_height, image_width])
