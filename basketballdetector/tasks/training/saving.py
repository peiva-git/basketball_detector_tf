import os


def save_models(model_wrapper):
    model_wrapper.model.save(
        filepath=os.path.join('out', 'models', 'Keras_v3', model_wrapper.model_name + '.keras')
    )
    model_wrapper.model.save(
        filepath=os.path.join('out', 'models', 'TF', model_wrapper.model_name), save_format='tf'
    )
    model_wrapper.model.save(
        filepath=os.path.join('out', 'models', 'HDF5', model_wrapper.model_name + '.h5'), save_format='h5'
    )