=======
EasyHPC
=======

A practical guide to your first High Performance Computing (HPC) project.

EasyHPC gathers what a new member of a research group usually learns over several months: how to collaborate on GitHub, how to work on a remote server, how to use GPUs, how to distribute training across them and how to solve differential equations with neural networks. Each page is short and comes with commands you can copy and run.

It started in 2022 as the shared notes of an undergraduate research course at Universidad Adolfo Ibáñez, Santiago, Chile, and is open to contributions.


Where to start
--------------

If you are new to all of this, read the pages in the order shown. If you already know your way around Git and Linux, jump to the topic you need.

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: GitHub
      :link: github
      :link-type: doc

      Fork a repository, work on a branch, open issues and send your first pull request.

   .. grid-item-card:: CI/CD
      :link: cicd
      :link-type: doc

      Write documentation in reStructuredText, build it with Sphinx and check every change with GitHub Actions.

   .. grid-item-card:: System administration
      :link: sysadmin
      :link-type: doc

      Linux shell, SSH, Jupyter on a remote server, GPUs, conda environments and Docker.

   .. grid-item-card:: Horovod
      :link: horovod
      :link-type: doc

      Train a TensorFlow model on several GPUs at once.

   .. grid-item-card:: DeepXDE
      :link: deepxde
      :link-type: doc

      Solve a Helmholtz equation with a physics-informed neural network.

   .. grid-item-card:: Neural network theory
      :link: nn
      :link-type: doc

      What a neural network computes, why the loss is a mean squared error and how PINNs compare with finite elements.


Contribute
----------

Found a typo, a broken command or a missing topic? :doc:`contributing` explains how to open an issue or send a pull request. :doc:`credits` lists the people who wrote EasyHPC.


.. toctree::
   :hidden:
   :caption: Collaborate

   github
   cicd

.. toctree::
   :hidden:
   :caption: Compute

   sysadmin
   horovod

.. toctree::
   :hidden:
   :caption: Physics-informed learning

   deepxde
   nn

.. toctree::
   :hidden:
   :caption: Project

   contributing
   credits
