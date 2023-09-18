# Basketball Detector

This repository contains the BasketballDetector implementation using a
classification approach instead of a segmentation one.
These results have been discarded since the segmentation approach gives a
much better result and is also way faster.

## Project requirements

This project uses `tensorflow==2.13.*`. Instructions on how to properly set up
a working environment can be found on the
[official page](https://www.tensorflow.org/install/pip).

## Project setup

To install all the required dependencies, either run:
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
