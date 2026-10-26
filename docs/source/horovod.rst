=======
Horovod
=======

`Horovod <https://horovod.ai>`__ trains a deep learning model on several GPUs, on one or several machines, with few changes to a single-GPU script.

.. note::

    The Horovod repository was archived in 2026 and its last release is v0.28.1 (June 2023). The examples below still illustrate how data-parallel training works. For new PyTorch projects, use the distributed training built into PyTorch (``torch.distributed`` and ``torchrun``).


Tutorials and documentation
---------------------------

#. `Getting started <https://horovod.ai/getting-started/>`__.
#. `Horovod with TensorFlow <https://horovod.readthedocs.io/en/stable/tensorflow.html>`__.
#. Towards Data Science: `Distributed Deep Learning with Horovod <https://towardsdatascience.com/distributed-deep-learning-with-horovod-2d1eea004cb2>`__.
#. The official tutorials::

    $ git clone https://github.com/horovod/tutorials


Run the Horovod examples on a GPU server
****************************************

The ``horovod`` Docker image comes with examples. Start it with::

    $ docker run --gpus all -it horovod/horovod

The ``examples`` directory has one subdirectory per framework::

    examples
    ├── adasum
    ├── elastic
    ├── keras
    ├── ...
    ├── tensorflow
    └── tensorflow2

To run the TensorFlow 2 benchmark on four GPUs::

    $ cd examples/tensorflow2
    $ CUDA_VISIBLE_DEVICES=0,1,2,3 horovodrun -np 4 -H localhost:4 python tensorflow2_synthetic_benchmark.py

If the terminal prints ``stddiag: Read -1``, see this `issue <https://github.com/horovod/horovod/issues/503>`__ to remove the warning.


Understanding the ``horovodrun`` command
****************************************

::

    CUDA_VISIBLE_DEVICES=0,1

This part makes only GPUs ``0`` and ``1`` visible to the command (see :doc:`sysadmin`).

::

    horovodrun -np 2 -H localhost:2 python tensorflow2_synthetic_benchmark.py

This part starts ``2`` processes (``-np 2``), both on the current machine (``-H localhost:2``). Each process trains on one GPU.


Use Horovod in your code
------------------------

This section adds Horovod to a TensorFlow 2 training script, following this `example <https://horovod.readthedocs.io/en/stable/tensorflow.html>`__.

#. Import Horovod::

    import tensorflow as tf
    import horovod.tensorflow as hvd

#. Initialize Horovod::

    hvd.init()

#. Pin one GPU to each process::

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

   ``set_memory_growth`` makes TensorFlow allocate GPU memory as needed, instead of reserving all of it at start. ``hvd.local_rank()`` is the index of the process on its machine, so process ``0`` uses the first visible GPU, process ``1`` the second one, and so on.

#. Define the model, the dataset, the loss and the optimizer. Scale the learning rate by the number of processes ``hvd.size()``::

    model = ...
    dataset = ...
    loss = tf.losses.SparseCategoricalCrossentropy()
    opt = tf.optimizers.Adam(0.001 * hvd.size())

#. Set up a checkpoint::

    checkpoint_dir = './checkpoints'
    checkpoint = tf.train.Checkpoint(model=model, optimizer=opt)

#. Define a training step::

    @tf.function
    def training_step(images, labels, first_batch):
        with tf.GradientTape() as tape:
            probs = model(images, training=True)
            loss_value = loss(labels, probs)

        tape = hvd.DistributedGradientTape(tape)

        grads = tape.gradient(loss_value, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))

        if first_batch:
            hvd.broadcast_variables(model.variables, root_rank=0)
            hvd.broadcast_variables(opt.variables(), root_rank=0)

        return loss_value

   ``hvd.DistributedGradientTape`` averages the gradients of all the processes before each update. After the first step, process ``0`` broadcasts its weights and optimizer state, so that all the processes start from the same values. The broadcast comes after the first step because the optimizer creates its variables during that step.

#. Train, printing and saving only from process ``0``::

    for batch, (images, labels) in enumerate(dataset.take(10000 // hvd.size())):
        loss_value = training_step(images, labels, batch == 0)

        if batch % 10 == 0 and hvd.rank() == 0:
            print('Step #%d\tLoss: %.6f' % (batch, loss_value))

    if hvd.rank() == 0:
        checkpoint.save(checkpoint_dir)

   Saving only from process ``0`` prevents the processes from overwriting each other's checkpoints.


Number of steps and learning rate
---------------------------------

Take a dataset of 60000 images and a batch size of 128. One epoch over the dataset on a single GPU takes

.. math::

    \left\lceil \frac{60000}{128} \right\rceil = 469 \text{ steps.}

With :math:`N` processes, each step processes :math:`N` batches at once, one per GPU, so the effective batch size is :math:`128 N` and one epoch takes about :math:`469 / N` steps. This is why the example divides the number of steps by ``hvd.size()`` and multiplies the learning rate by ``hvd.size()``.
