Python Programming
==================

Horovod
---------------

Example of script using horovod::

$ CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 horovodrun -np 2 -H localhost:2 python pytorch_mnist.py 


Getting started: Go to `this link <https://horovod.ai/getting-started/>`_.

`Horovod with TensorFlow <https://horovod.readthedocs.io/en/stable/tensorflow.html>`_.

Towards Data Science: `Distributed Deep Learning with Horovod <https://towardsdatascience.com/distributed-deep-learning-with-horovod-2d1eea004cb2>`_.


Tutorials
---------------

Tutorials for Horovod: ::

$ git clone https://github.com/horovod/tutorials

