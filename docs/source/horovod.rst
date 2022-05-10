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

    horovodrun -np 2 -H localhost:2 python tensorflow2_synthetic_benchmark.py

This part explicitly calls horovodrun with ``2`` gpus in the localhost, this case is assuming that you are 
working on only one machine.


.. Later on in this part we will add the parallel to DeepXDE.

Use horovod
-----------
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


Simple example
--------------

For this tutorial we will be using the `keras example <https://www.tensorflow.org/datasets/keras_example>`_ from the
official TensorFlow documentation.

1. Horovod configuration and definition of the dataset ::

    import tensorflow as tf
    import tensorflow_datasets as tfds
    import horovod.tensorflow.keras as hvd

    hvd.init()

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label

    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

2. If you want to know which is the size of this dataset you can run the following before you start training::

    print(f"Length: {len(ds_train)}")

3. Now define the ``model``::

    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
    ])

4. Now we will be using the ``compile`` and ``fit`` method to train our data::

    opt = tf.optimizers.Adam(0.001 * hvd.size())
    opt = hvd.DistributedOptimizer(opt)


    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        experimental_run_tf_function=False,
    )

    callbacks = [
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    ]

    if hvd.rank() == 0:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))

    model.fit(
        ds_train,
        callbacks=callbacks,
        epochs=6,
        validation_data=ds_test,
        verbose=1 if hvd.rank() == 0 else 0
    )


Note
******

.. todo::
    
    Add documentation about tf.GradientTape(), etc.

We will compare this with the use of a ``training_step`` function and the ``for loop``
to train our model.

Observations
**************

The original code would ran on the CPU, but with this implementation it will run on the GPU(s).

We are working with a dataset of 60000 images, with 6 epochs and a batch size of 128. And 
therefore 469 number of iterations.

.. math:: 
    \frac{\mbox{number of total data}}{\mbox{batch size}} = \mbox{number of iterations} \rightarrow \frac{60000}{128} = 469

In this case, the dataset is finite so we can't decide how many ``steps_per_epoch`` we want. 

Infinite amount of data
*************************

If we would have an *infinite* amount of data, we would define the ``steps_per_epoch`` we would want in the ``fit`` method.

.. math:: 
    \mbox{steps per epoch} = \frac{\mbox{quantity of desired steps per epoch}}{\mbox{number of gpus}}

And the quantity number of data that our model will take to train will be in this form. 

.. math:: 
    \mbox{number of total data} = \mbox{number of iterations} \cdot {\mbox{batch size}}

And don't forget to include the quantity of epochs in the ``fit`` method.