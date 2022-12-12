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
| Fifty-One Library | data gathering / data annotation | FiftyOne is a very useful library for visualization and annotation of your data | [Quick start](data_gathering/FiftyOne/FiftyOne.py) |
| SimpleImage Library | data gathering | A keyword downloader for Google images | [Quick start](data_gathering/SimpleImage/SimpleImage.py) |
| LabelMe library | data annotation | LabelMe is a graphical image annotation tool. It is written in Python and uses Qt for its graphical interface. | [Quick start](data_annotation/LabelMe/README.md) |
| CLoDSA | data augmentation | CLoDSA is an open-source image augmentation library for object classification, localization, detection, semantic segmentation and instance segmentation. | [Quick start](data_augmentation/CLoDSA/CLoDSA.ipynb) |
| Detectron2 | cv model (multiple tasks) | Detectron2 is Facebook AI Research's next generation library that provides state-of-the-art detection and segmentation algorithms. It is the successor of Detectron and maskrcnn-benchmark. It supports a number of computer vision research projects and production applications in Facebook. | [Quick start](cv_algorithms/Detectron_2/Detectron_2.ipynb) |
| Mask RCNN | cv model (detection/segmentation tasks) | Mask R-CNN is a state of the art model for instance segmentation, developed on top of Faster R-CNN. | [Quick start](cv_algorithms/MaskRCNN/MaskRCNN.ipynb) |
| Yolov5 | cv model (detection task) | YOLOv5 is an open-source project that consists of a family of object detection models and detection methods based on the YOLO model pre-trained on the COCO dataset. It is maintained by Ultralytics and represents the organization's open-source research into the future of Computer Vision works. | [Quick start](cv_algorithms/YOLOv5/YOLOv5.ipynb) |
| Human Pose | cv algorithms for classification of human pose| It uses [OpenPose]("https://github.com/CMU-Perceptual-Computing-Lab/openpose") which is the first real-time multi-person system to jointly detect human body, hand, facial, and foot keypoints (in total 135 keypoints) on single images, [MoveNet]("https://www.tensorflow.org/hub/tutorials/movenet") which is an ultra fast and accurate model that detects 17 keypoints of a body and a custom tensorflow model. | [Quick start](cv_algorithms/Human_pose_classification/Human_pose_classification.ipynb) |

## Contributing

We welcome contributions and ideas for this project. Please review our [contribution guidelines](CONTRIBUTING.md) before contributing.

## Reference papers / repos
- [Facebook AI Research Detectron2](https://github.com/facebookresearch/detectron2)
- [Mask RCNN](https://arxiv.org/abs/1703.06870)
- [Human Pose guide]("https://www.tensorflow.org/lite/tutorials/pose_classification")
- [MoveNet]("https://www.tensorflow.org/hub/tutorials/movenet")
- [OpenPose]("https://cmu-perceptual-computing-lab.github.io/openpose/web/html/doc/md_doc_00_index.html")
