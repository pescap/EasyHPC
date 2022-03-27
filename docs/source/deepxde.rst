DeepXDE
=======

To do. Explain the theory for inverse problems, and add references.

Installation 
============


**Requirements**

To run **DeepXDE** in your local machine or in a computer cluster is necessary install **TensorFlow** as a backend library.
In order to do that, first you need to install **TensorFlow** using the command line: ::

$ pip install TensorFlow  #Using pip as a installer
$ conda install TensorFlow #Using conda as a installer


The next step is to **install the DeepXDE library** with one of these commands: ::

$ pip install deepxde #Using pip as installer
$ conda install deepxde #Using conda as installer

Once TensorFlow and DeepXDE were installed, you can check if the installation was successful running a IPython or Jupyter session and importing the library on it.


**How to run DeepXDE examples**

The first step is to clone the library repository in your computer cluster using git: ::

$ git clone https://github.com/lululxvi/deepxde.git

Second step, change Python PATH to the library directory: ::

$ export PYTHONPATH=$PYTHONPATH:/root/shared/deepxde

One time the repository was clone and the PATH was updated, you are ready to run some library examples that are located in the examples directory.




Bibliography
------------

- DeepXDE: A Deep Learning Library for solving differential equations, Lu, Lu and Meng, Xuhui and Mao, Zhiping and Karniadakis, George Em, SIAM Review (2021) [`link <https://epubs.siam.org/doi/pdf/10.1137/19M1274067>`_]
- Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations, M. Raissi and P. Perdikaris and G.E. Karniadakis, Journal of Computational Physics (2019) [`link <https://www.sciencedirect.com/science/article/pii/S0021999118307125>`_]