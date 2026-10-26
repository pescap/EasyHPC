=====================
System Administration
=====================

This page covers the everyday tools you need on a Linux server: the shell, remote connections, GPUs, Python environments and containers.


Linux command line for beginners
--------------------------------

File commands
*************

::

    $ ls                     # List the files of the current directory
    $ cd                     # Go to your home directory
    $ cd folder_example      # Enter the directory folder_example
    $ cd ..                  # Go to the parent directory
    $ pwd                    # Print the path of the current directory
    $ mkdir folder_example   # Create the directory folder_example
    $ cat file_name          # Print the content of file_name
    $ rm file_name           # Delete file_name
    $ rm -r folder_example   # Delete folder_example and everything inside it

``rm`` does not use a trash bin: deleted files cannot be recovered.


System information
******************

::

    $ htop                   # Show processes, CPU and memory usage (press q to quit)
    $ df -h                  # Show the free space of each disk
    $ du -sh folder_example  # Show the size of folder_example
    $ date                   # Show the current date and time


Remote servers and Jupyter
--------------------------

Connect via SSH to the server ``server_address`` with the user name ``user``::

    $ ssh user@server_address

To use Jupyter on the server from the browser of your computer, connect with port forwarding on port ``8888``::

    $ ssh -L 8888:localhost:8888 user@server_address

Once logged into the server, start Jupyter without opening a browser there::

    $ jupyter lab --no-browser --port 8888

Jupyter prints a URL starting with ``http://localhost:8888``. Copy and paste it into the browser of your computer.

Sometimes the port stays in use after you close Jupyter. To see which process holds it::

    $ lsof -i :8888

The second column of the output is the process identifier, ``PID``. Stop the process with::

    $ kill PID

If the process does not stop, force it with ``kill -9 PID``.


GPU programming
---------------

To check the GPUs of the machine, their memory and the processes using them, run::

    $ nvidia-smi

The environment variable ``CUDA_VISIBLE_DEVICES`` sets which GPUs a program can see. To run a script ``my_program.py`` on GPUs ``0`` and ``2`` only::

    $ CUDA_VISIBLE_DEVICES=0,2 python my_program.py

To run it on the CPU only, hide all the GPUs::

    $ CUDA_VISIBLE_DEVICES="" python my_program.py


Conda
-----

Conda manages packages and environments. Each environment has its own Python version and packages, which avoids conflicts between projects.


Install conda
*************

`Miniforge <https://github.com/conda-forge/miniforge>`__ is a minimal conda installer that uses the free conda-forge channel. On Linux and macOS, download and run the installer for your system::

    $ curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    $ bash Miniforge3-$(uname)-$(uname -m).sh

Follow the instructions of the installer. When it asks whether to initialize conda, answer ``yes``, then restart your terminal. If the ``conda`` command is not found afterwards, run::

    $ ~/miniforge3/bin/conda init

and restart the terminal again.


Manage environments
*******************

Create an environment called ``new_env`` with a given Python version::

    $ conda create -n new_env python=3.12

List your environments::

    $ conda env list

Activate the environment, and deactivate it when you are done::

    $ conda activate new_env
    $ conda deactivate

Install a package in the active environment, with conda or with pip::

    $ conda install numpy
    $ pip install numpy

Packages installed in an environment are only available in that environment. See the `conda documentation <https://docs.conda.io/projects/conda/en/latest/index.html>`__ for more.


Docker
------

Docker runs software inside containers that ship with all their dependencies. With the `NVIDIA Container Toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>`__ installed, containers can use the GPUs of the machine.

To start a DeepXDE container with all the GPUs, the current directory mounted in ``/root/shared`` and Jupyter's port forwarded::

    $ docker run --gpus all -v $(pwd):/root/shared -w /root/shared -p 8888:8888 pescapil/deepxde:latest

To use your own copy of DeepXDE inside the container instead of the installed one, add it to the ``PYTHONPATH``::

    $ export PYTHONPATH=$PYTHONPATH:/root/shared/deepxde
