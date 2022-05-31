=======
Horovod
=======

Tutorials and documentation
---------------------------

1. Getting started: Go to `this link <https://horovod.ai/getting-started/>`_.

2. How to implement `Horovod with TensorFlow <https://horovod.readthedocs.io/en/stable/tensorflow.html>`_.

3. Towards Data Science: `Distributed Deep Learning with Horovod <https://towardsdatascience.com/distributed-deep-learning-with-horovod-2d1eea004cb2>`_. 

4. Some tutorials for Horovod are available here: ::

	$ git clone https://github.com/horovod/tutorials



Run Horovod examples on a GPU cluster
***************************************

The ``horovod`` Docker image comes with examples. Run ::

	$ nvidia-docker run -it horovod/horovod

The `examples` directory comes with an example directory per backend ::

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

Understanding the ``horovodrun`` command
******************************************

1. ::

    CUDA_VISIBLE_DEVICES="0,1"

This part allows you to pass certain gpus to the ``horovodrun`` command, in this case we are using the gpu 
``0`` and ``1``.

2. ::

    horovodrun -np 2 -H localhost:2 python tensorflow2_synthetic_benchmark.py

This part explicitly calls horovodrun with ``2`` gpus in the localhost, this case is assuming that you are 
working on only one machine.


.. Later on in this part we will add the parallel to DeepXDE.

Use horovod
-----------
In this section we will implement Horovod to a TensorFlow V2 code from this `example <https://horovod.readthedocs.io/en/stable/tensorflow.html>`_.

1. Import horovod ::

    import tensorflow as tf
    import horovod.tensorflow as hvd

2. Initialize horovod ::

    hvd.init()

3. Pin your GPUS (given the ``CUDA_VISIBLE_DEVICES`` gpus) ::

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

The ``for`` part of this code allows you to tell the program to use variable memory and not allocating the entire 
GPU VRAM.

4. After setting your dataset and model ::

    model = ...
    dataset = ...
    opt = tf.optimizers.Adam(0.001 * hvd.size())
    loss = tf.losses.SparseCategoricalCrossentropy()

5. Set up the function ::
    
    checkpoint_dir = './checkpoints'
    checkpoint = tf.train.Checkpoint(model=model, optimizer=opt)

6. Function ::

    def training_step(images, labels, first_batch):
        with tf.GradientTape() as tape:
            probs = mnist_model(images, training = True)
            loss_value = loss(labels, probs)
        
        # Horovod: add Horovod Distributed GradientTape.
        tape = hvd.DistributedGradientTape(tape)

        grads = tape.gradient(loss_value, mnist_model.trainable_variables)
        opt.apply_gradients(zip(grads, mnist_model.trainable_variables))

        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        #
        # Note: broadcast should be done after the first gradient step to ensure optimizer
        # initialization.
        if first_batch:
            hvd.broadcast_variables(mnist_model.variables, root_rank = 0)
            hvd.broadcast_variables(opt.variables(), root_rank = 0)

        return loss_value

7. Run the code ::

    # Horovod: adjust number of steps based on number of GPUs.
    for batch, (images, labels) in enumerate(dataset.take(10000 // hvd.size())):
        loss_value = training_step(images, labels, batch == 0)

        if batch % 10 == 0 and hvd.local_rank() == 0:
            print('Step #%d\tLoss: %.6f' % (batch, loss_value))

    # Horovod: save checkpoints only on worker 0 to prevent other workers from
    # corrupting it.
    if hvd.rank() == 0:
        checkpoint.save(checkpoint_dir)

Observations
**************

Every code would run on the CPU, but with this implementation it will run on the GPU(s).

Example: If we are working with a dataset of 60000 images, with 6 epochs and a batch size of 128. And 
therefore 469 number of iterations.

.. math:: 
    \frac{\mbox{number of total data}}{\mbox{batch size}} = \mbox{number of iterations} \rightarrow \frac{60000}{128} = 469

In this case, the dataset is finite so we can't decide how many ``steps_per_epoch`` we want. 

Infinite amount of data
*************************

If we would have an *infinite* amount of data:

.. math:: 
    \mbox{steps per epoch} = \frac{\mbox{quantity of desired steps per epoch}}{\mbox{number of gpus}}

And the quantity number of data that our model will take to train will be in this form. 

.. math:: 
    \mbox{number of total data} = \mbox{number of iterations} \cdot {\mbox{batch size}}