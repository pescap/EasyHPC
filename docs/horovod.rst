Python Programming
==================

Horovod
---------------

Example of script using horovod::

$ CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 horovodrun -np 2 -H localhost:2 python pytorch_mnist.py 

