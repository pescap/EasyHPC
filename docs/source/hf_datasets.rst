=========================
Hugging Face datasets
=========================

The `Hugging Face Hub <https://huggingface.co/datasets>`__ hosts thousands of public datasets, and the `datasets <https://huggingface.co/docs/datasets/index>`__ library loads them in a few lines. This page shows how to use them on a server or a cluster, where the home directory is small and compute nodes often have no internet access.


Install
-------

In your environment (see :doc:`sysadmin`)::

    $ pip install datasets


Load a dataset
--------------

::

    from datasets import load_dataset

    dataset = load_dataset("rajpurkar/squad")
    print(dataset)
    print(dataset["train"][0])

The first call downloads the files and converts them to the Arrow format. The next calls reuse the local copy.

To feed the data to PyTorch, return tensors instead of Python lists::

    train = dataset["train"].with_format("torch")

For datasets too large for your disk, ``streaming=True`` reads the examples while downloading, without storing the whole dataset::

    dataset = load_dataset("rajpurkar/squad", split="train", streaming=True)
    for example in dataset:
        ...


Move the cache out of your home
-------------------------------

By default, everything goes to ``~/.cache/huggingface``, which quickly fills the quota of a home directory. Point the cache to a larger file system, for example a scratch directory, by adding this line to your ``~/.bashrc``::

    $ export HF_HOME=/scratch/user/.cache/huggingface

``HF_HOME`` moves both the downloaded files and the converted datasets. See `Cache management <https://huggingface.co/docs/datasets/cache>`__ for finer control.


Private and gated datasets
--------------------------

Some datasets require you to log in and accept their terms on the website. Create an access token in your Hugging Face settings, then log in once on the server with the ``hf`` command, which comes with the ``huggingface_hub`` package::

    $ hf auth login


Compute nodes without internet
------------------------------

On many clusters only the login node can reach the internet. Download the dataset there, into a directory of your choice::

    $ hf download rajpurkar/squad --repo-type dataset --local-dir /scratch/user/data/squad

In your job, load it from that directory and tell the libraries not to try the network::

    $ export HF_HUB_OFFLINE=1

::

    from datasets import load_dataset

    dataset = load_dataset("/scratch/user/data/squad")


Save a processed dataset
------------------------

After tokenizing or filtering a dataset, save the result to avoid recomputing it in every job::

    dataset.save_to_disk("/scratch/user/data/squad_processed")

and load it later with::

    from datasets import load_from_disk

    dataset = load_from_disk("/scratch/user/data/squad_processed")
