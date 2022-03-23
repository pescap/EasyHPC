Horovod
=======

Example of script using horovod::

$ CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 horovodrun -np 2 -H localhost:2 python pytorch_mnist.py 


Getting started: Go to `this link <https://horovod.ai/getting-started/>`_.

`Horovod with TensorFlow <https://horovod.readthedocs.io/en/stable/tensorflow.html>`_.

Towards Data Science: `Distributed Deep Learning with Horovod <https://towardsdatascience.com/distributed-deep-learning-with-horovod-2d1eea004cb2>`_.


Tutorials
---------

Some tutorials for Horovod are available here: ::

	$ git clone https://github.com/horovod/tutorials


Run Horovod examples on a GPU cluster
-------------------------------------

The ``horovod`` Docker image comes with examples. Run ::

	$ nvidia-docker run -it horovod/horovod

The `examples` directory comes with an example directory per backend

::

    examples
    ├── adasum
    ├── elastic
    ├── keras
    ├── ...
    ├── tensorflow
    └── tensorflow2

If you choose the `tensorflow2` backend ::

	$ cd tensorflow2
	$ CUDA_VISIBLE_DEVICES=0,1,2,3 horovodrun -np 4 -H localhost:4 python tensorflow2_synthetic_benchmark.py

If the terminal flushes ``stddiag: Read -1``, refer to this `issue <https://github.com/horovod/horovod/issues/503>`_ to remove the warning.