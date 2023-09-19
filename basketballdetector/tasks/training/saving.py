import os


def save_models(model_wrapper):
    """
    Save the provided model in the Keras, TF and HDF5 formats.
    The models will be saved in the `out/models` directory.
    :param model_wrapper: The model to be saved. Must be an instance of one of the models provided in the
    `basketballdetector.models` package.
    :return: None
    """
    model_wrapper.model.save(
        filepath=os.path.join('out', 'models', 'Keras_v3', model_wrapper.model_name + '.keras')
    )
    model_wrapper.model.save(
        filepath=os.path.join('out', 'models', 'TF', model_wrapper.model_name), save_format='tf'
    )
    model_wrapper.model.save(
        filepath=os.path.join('out', 'models', 'HDF5', model_wrapper.model_name + '.h5'), save_format='h5'
    )
