# Basketball Detector

:basketball: **BasketballDetector** is a deep-learning based tool
that enables automatic ball detection in basketball broadcast videos.

This repository contains the BasketballDetector implementation using a
classification approach instead of a segmentation one.
It also contains some of the first attempts to address the problem
with simple segmentation models implemented in tensorflow from scratch.

Please note that **this work has been discarded** in favor of the
segmentation approach using SOTA real-time segmentation models, which
proved more accurate and way faster.

## Project requirements

This project uses `tensorflow==2.13.*`. Instructions on how to properly set up
a working environment can be found on the
[official page](https://www.tensorflow.org/install/pip).

Alternatively, you can simply import and use the same 
[conda](https://docs.conda.io/projects/conda/en/latest/index.html)
environment that was used during development.
Using the provided [conda environment file](conda/tf-environment.yml) run:
```shell
conda create --name myenv --file tf.environment.yml
```
Don't forget to set up the required environment variables as well:
```shell
CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH
```

You can automatize the process of adding the environment variables
to execute automatically each time you activate your
conda environment by running the following commands:
```shell
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

## Project setup

If you're using the provided conda environment, you can ignore these steps.
Otherwise, to install all the required dependencies, either run:
```shell
python -m pip install -r requirements.txt
```
Or:
```shell
python -m pip install .
```

If you want to install the project in development mode, instead you can run:
```shell
python -m pip install -v -e .
```

More information about what development mode is can be found
[here](https://setuptools.pypa.io/en/latest/userguide/development_mode.html).
