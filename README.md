# Computer Vision

## Introduction

This repository contains examples and best practices for building Computer Vision systems, provided as Jupyter notebooks.

In [cv_algorithms](cv_algorithms), a number of utilities are included to facilitate standard tasks like importing datasets and models in the formats required by various algorithms. For self-study and customisation in your own applications, there are implementations of a number of cutting-edge algorithms available.

For more information regarding the theoretical part of Computer Vision please follow the guide [here](cv_algorithms/README.md).

## Getting Started

For additional information on configuring your system locally, please refer to the [setup guide](SETUP.md).

For the libraries/models which have the indication of "local machine" in the table below the installation has been tested using
- Python version 3.8 and [venv](https://docs.python.org/3/library/venv.html), or [conda](https://docs.conda.io/projects/conda/en/latest/glossary.html?highlight=environment#conda-environment)

The package and its dependencies should be installed in a clean environment (such as
[conda](https://docs.conda.io/projects/conda/en/latest/glossary.html?highlight=environment#conda-environment) or [venv](https://docs.python.org/3/library/venv.html)).


## Algorithms

The CV algorithms or libraries that are currently offered in the repository are listed in the table below. Under the Example column, you can find links to code. In addition, under the type column you can find a label on whether the example refers to data gathering, data annotation, data augmentation or a cv model (with the specific cv task). 


| Algorithm / Library | Type | Description | Example |
|-----------|------|-------------|---------|
| Fifty-One Library | data gathering / data annotation | FiftyOne is a very useful library for visualization and annotation of your data | [Quick start](data_gathering/FiftyOne_(for_M1).py) |
| SimpleImage Library | data gathering | A keyword downloader for Google images | [Quick start](data_gathering/SimpleImage.py) |
| LabelMe library | data annotation | LabelMe is a graphical image annotation tool. It is written in Python and uses Qt for its graphical interface. | [Quick start](data_annotation/LabelMe.py) |
| CLoDSA | data augmentation | CLoDSA is an open-source image augmentation library for object classification, localization, detection, semantic segmentation and instance segmentation. | [Quick start](data_augmentation/CLODSA.ipynb) |
| Detectron2 | cv model (multiple tasks) | Detectron2 is Facebook AI Research's next generation library that provides state-of-the-art detection and segmentation algorithms. It is the successor of Detectron and maskrcnn-benchmark. It supports a number of computer vision research projects and production applications in Facebook. | [Quick start](examples/Detectron_2.ipynb) |
| Mask RCNN | cv model (detection/segmentation tasks) | Mask R-CNN is a state of the art model for instance segmentation, developed on top of Faster R-CNN. | [Quick start](examples/MaskRCNN.ipynb) | G
| Yolov5 | cv model (detection task) | YOLOv5 is an open-source project that consists of a family of object detection models and detection methods based on the YOLO model pre-trained on the COCO dataset. It is maintained by Ultralytics and represents the organization's open-source research into the future of Computer Vision works. | [Quick start](examples/YOLOv5.ipynb) |

## Contributing

We welcome contributions and ideas for this project. Please review our [contribution guidelines](CONTRIBUTING.md) before contributing.

## Reference papers / repos
- [Facebook AI Research Detectron2](https://github.com/facebookresearch/detectron2)
- [Mask RCNN](https://arxiv.org/abs/1703.06870)
