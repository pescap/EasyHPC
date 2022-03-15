Python Programming
==================

Conda environments
------------------
To help you work with differents versions of Python and different packages/libraries to avoid compatibility issues.

1. Installing Anaconda.
    Unix:
       1. Download the latest version of `Conda <https://www.anaconda.com/products/individual>`_.
       2. Run::

            $ bash <name of file downloaded>.sh

       3. Follow the instructions on the installer screens.
       4. **Remember** to accept, at the end of everything, adding **Conda** to **PATH**.
       5. Restart your terminal window.

    MacOS:
       1. Download the latest version of `Conda <https://www.anaconda.com/products/individual>`_.
       2. Open the ``.pkg`` file.
       3. Follow the instructions on the installer screens.
       4. **Remember** to accept, at the end of everything, adding **Conda** to **PATH**.
       5. Restart the terminal window.

    **Note**: If conda command doesn't work, run::
        
        $ conda init

2. Creating/Managing your environments.
    1. To create an environment with a specific version::
        
        $ conda create -n new_env python=<version_number>

    2. Checking the list of your environments::
        
        $ conda env list

    3. Activating/Deactivating an environment::
        
        $ conda activate/deactivate new_env

    4. Now you can install packages with either the command pip or conda.

Remember that any packages/libraries installed on a specific conda environment are retained there, environments do not share installed packages.
    

Horovod
-------

Example of script using horovod::

$ CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 horovodrun -np 2 -H localhost:2 python pytorch_mnist.py 


Getting started: Go to `this link <https://horovod.ai/getting-started/>`_.

`Horovod with TensorFlow <https://horovod.readthedocs.io/en/stable/tensorflow.html>`_.

Towards Data Science: `Distributed Deep Learning with Horovod <https://towardsdatascience.com/distributed-deep-learning-with-horovod-2d1eea004cb2>`_.


Tutorials
---------

Tutorials for Horovod: ::

$ git clone https://github.com/horovod/tutorials

