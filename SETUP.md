# Setup guide

The following platforms are covered in this document's setup instructions for all the dependencies needed to run the notebooks in this repository:

* Local (Linux, MacOS or Windows) 
* Docker container

## Table of Contents

  - [Compute environments](#compute-environments)
  - [Setup guide for Local](#setup-guide-for-local)
    - [Requirements](#requirements)
    - [Dependencies setup](#dependencies-setup)
    - [Using a virtual environment](#using-a-virtual-environment)
    - [Register the environment as a kernel in Jupyter](#register-the-environment-as-a-kernel-in-jupyter)
  - [Setup guide for Docker](#setup-guide-for-docker)
  - [Setup guide for making a release](#setup-guide-for-making-a-release)

Different computational needs apply depending on the Computer Vision system type and the notebook that needs to be run.
Currently, this repository supports **Python CPU** and **Python GPU**

## Setup guide for Local

The computer vision algorithms can be applied in a variety of ways.

Building a docker image and utilizing its features is one method with a [docker container](#setup-guide-for-docker).

Another option is to directly execute every computer vision algorithm from a local copy of the source code. Installing all required Anaconda and PyPI dependencies is required for this. To learn how to do this, refer to [this guide](conda.md).

### Requirements

* A machine running Linux, MacOS or Windows
* An optional requirement is Anaconda with Python version == 3.10
 * To setup on your local machine, [Miniconda](https://docs.conda.io/en/latest/miniconda.html) is a quick way to get started.

 Alternatively a [virtual environment](#using-a-virtual-environment) can be used instead of Anaconda.

### Dependencies setup

If using Conda, make sure that Anaconda and the package manager Conda are both up to date before installing the dependencies:

```{shell}
conda update conda -n root
conda update anaconda        # use 'conda install anaconda' if the package is not installed
```

If using venv or virtualenv, see [these instructions](#using-a-virtual-environment).

### Using a virtual environment


### Register the environment as a kernel in Jupyter

You can register the conda or virtual environment to appear as a kernel in the Jupyter notebooks. After activating the environment (`my_env_name`) do

    python -m ipykernel install --user --name my_env_name --display-name "Python (my_env_name)"

## Setup guide for Docker

To make setting up for various environments simpler, a [Dockerfile](tools/docker/Dockerfile) is supplied to build images of the repository. On your computer, you must have [Docker Engine](https://docs.docker.com/install/) installed.

To build and execute images for various environments, refer to the Docker [README](tools/docker/README.md) for complete instructions.

You can then open the Jupyter notebook server at http://localhost:8888

## Setup guide for making a release

