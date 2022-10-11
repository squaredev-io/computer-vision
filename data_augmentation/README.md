# Data augmentation

## Introduction

In data analysis, procedures called "data augmentation" are used to expand the amount of data by adding slightly changed versions of either existing data or brand-new synthetic data that is derived from existing data. It serves as a regularizer and aids in lowering overfitting while a machine learning model is being trained [[1]](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0197-0). It is strongly connected to data analysis oversampling.
If a dataset is relatively tiny, adding rotation, mirroring, and other enhancements may still not be sufficient to solve a particular issue. The creation of brand-new, synthetic pictures using a variety of approaches, such as the usage of generative adversarial networks to produce fresh synthetic images for data augmentation, is another option [[2]](https://ieeexplore.ieee.org/document/9199968). Additionally, when moving from photos created in virtual settings to real-world data, image recognition algorithms demonstrate progress.

## CLoDSA

For object categorization, localisation, detection, semantic segmentation, and instance segmentation, CLoDSA is an [open-source](https://github.com/joheras/CLoDSA) image augmentation package. It supports a wide range of augmentation methods and makes it simple for the user to mix them. Similar to movies or z-stacks, it may also be used to organize lists of photographs.

In this [study](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-2931-1), it is described a general approach that may be used to automatically enlarge a multi-dimensional picture dataset used for instance segmentation, semantic segmentation, localisation, or classification. The open-source software CLoDSA has been updated to include the augmentation approach described in this work. To demonstrate the advantages of employing CLoDSA, we have used this library to increase the precision of models for stomata recognition, automated segmentation of brain structures, and Malaria parasite categorization.

The features it provides are:

- There are several enhancement methods offered.
- Techniques for augmentation can be used for instance segmentation, semantic segmentation, localisation, detection, and object categorization. Techniques for augmentation can also be used with collections of photos (a stack can be, for instance, a z-stack of images acquired with a microscopy or a video).
- A Java wizard for library configuration.
- For reading the datasets, there are various input modes.
- To create the augmented dataset, there are various output modes.
- For object classification, localisation, detection, semantic segmentation, and instance segmentation in both images and videos, it is simple to incorporate additional augmentation techniques.