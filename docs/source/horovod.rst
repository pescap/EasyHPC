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
-------------------------------------

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
----------------------------------------

1. ::

    CUDA_VISIBLE_DEVICES="0,1"

This part allows you to pass certain gpus to the ``horovodrun`` command, in this case we are using the gpu 
``0`` and ``1``.

2. ::

    horovodrun -np 2 -H localhost:2

This part explicitly calls horovodrun with ``2`` gpus in the localhost, this case is assuming that you are 
working on only one machine.

3. ::

    python tensorflow2_synthetic_benchmark.py

And finally you use python to run the wanted file.


.. Later on in this part we will add the parallel to DeepXDE.

Implementing horovod
--------------------
In this section we will implement Horovod to a TensorFlow v2 Keras code.

1. Import horovod ::

    import tensorflow as tf
    import horovod.tensorflow.keras as horovod

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

5. Add the Horovod DistributedOptimizer ::

    opt = hvd.DistributedOptimizer(opt)

6. Now, specify ``experimental_run_tf_function=false`` ::

    mnist_model.compile(loss=tf.losses.SparseCategoricalCrossentropy(),
                    optimizer=opt,
                    metrics=['accuracy'],
                    experimental_run_tf_function=False)
    
This ensure TensorFlow uses hvd.DistributedOptimizer().

7. Setting callbacks ::

    callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0),]

This will broadcast the initial variable states from rank 0 to every processes. 
Just like the it is explained in the `official documentation <https://horovod.readthedocs.io/en/stable/keras.html>`_ 
this is necessary to ensure consistent intialization of all workers.

8. Save checkpoints only on worker 0 ::

    if hvd.rank() == 0:
        callbacks.append(keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))

9. Fitting the model ::

    model.fit(dataset,
          steps_per_epoch=500 // hvd.size(),
          callbacks=callbacks,
          epochs=24,
          verbose=1 if hvd.rank() == 0 else 0)

Remember if the verbose is needed, assign it to ``1`` if there is only one GPU, else is ``0``.