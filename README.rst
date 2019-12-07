Overview
========

This project introduces an approach for trying to solve the Traveling Salesman Problem with graph convolutional and sequence-to-sequence networks. The first step relies mostly on an approach followed by C. K. Joshi, T. Laurent, and X. Bresson aiming at building efficient TSP graph representations with Graph ConvNets, on top of which we tried to implement a structure described as Pointer Network by O. Vinyals, M. Fortunato, N. Jaitly. This project was conducted in the context of a Deep Learning course taught by Pr. Iddo Drori from the Columbia University Computer Science Department.

Main parts of our code and notebooks have been taken and inspired from previous work from C. K. Joshi, T. Laurent, and X. Bresson, which could be found on the following repository https://github.com/chaitjo/graph-convnet-tsp#installation .


Usage
=====

Installation
------------

The below step-by-step guide for local installation using a Terminal (Mac/Linux) or Git Bash (Windows) has been imported from 

.. code-block:: pyhon

   # Install [Anaconda 3](https://www.anaconda.com/) for managing Python packages and environments.
   curl -o ~/miniconda.sh -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
   chmod +x ~/miniconda.sh
   ./miniconda.sh
   source ~/.bashrc

   # Clone the repository. 
   git clone https://github.com/chaitjo/graph-convnet-tsp.git
   cd graph-convnet-tsp

   # Set up a new conda environment and activate it.
   conda create -n gcn-tsp-env python=3.6.7
   source activate gcn-tsp-env

   # Install all dependencies and Jupyter Lab (for using notebooks).
   conda install pytorch=0.4.1 cuda90 -c pytorch
   conda install numpy==1.15.4 scipy==1.1.0 matplotlib==3.0.2 seaborn==0.9.0 pandas==0.24.2 networkx==2.2 scikit-learn==0.20.2 tensorflow-gpu==1.12.0 tensorboard==1.12.0 Cython
   pip3 install tensorboardx==1.5 fastprogress==0.1.18
   conda install -c conda-forge jupyterlab

