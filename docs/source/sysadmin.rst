System Administration
======================

Linux command line for beginners
--------------------------------

Here goes a short list of useful commands:

**File commands**::


$ ls # Show current directory
$ cd # Go to the root directory
$ cd example_directory # Changing to another directory 
$ cd - # Go back to the previous directory
$ mkdir newproject # Create a new directory 
$ rm file  # Delete file
$ rm -r trashfolder # Delete directory
$ pwd # Show the current path

**System info**::


$ htop # Check memory usage and processes
$ df # Show disk usage
$ date # Show the current date and time

Users management (sudo)
-----------------------

Create a new user with username ``user``::

$ sudo adduser user
$ sudo passwd -e user # user will be asked to set new password

Then, one can define groups privileges. For example, to create an ``anaconda`` group that gives read access to path ``/opt/conda``::

$ sudo groupadd anaconda
$ sudo chgrp -R anaconda /opt/conda
$ sudo usermod -aG anaconda user

List sudoers: ::

$ grep -Po '^sudo.+:\K.*$' /etc/group

Add or remove ``user`` to sudo::

$ sudo usermod -aG sudo user

Install Oh My Zsh
-----------------

Run: ::

 $ sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

Cluster and Jupyter Notebook
----------------------------

Connect via ssh to ``ip_address`` with username ``user``: ::

 $ ssh user@ip_address

Connect via ssh to the server, enabling port forwarding with port ``8888`` using::

$ ssh -L 8888:localhost:8888 user@ip_address

Once logged into the server, simply run::

$ jupyter-notebook

Then, copy-paste the url to run the jupyter-notebook in your bowser. Sometimes, the port-forwarding keeps working albeit having closed the jupyter session. You can check port forwarding by running::

$ lsof -i:8888

This command shows you the active process identities ``PID``. To kill the process, type::

$ kill -9 PID

with ``PID`` the process number.

GPU programming
---------------

To check GPUs usage, run::

$ nvidia-smi

Available GPUs are defined with global variable ``CUDA_VISIBLE_DEVICES``. To run a script ``my_program`` with specific GPUs (e,g, ``0,2``), run::

$ CUDA_VISIBLE_DEVICES="0,2" my_program

Example in CPU mode (no visible GPUs), to run a python script called ``dl.py``::

$ CUDA_VISIBLE_DEVICES="" python dl.py


Docker
------
 
To run a DeepXDE container, run: ::

$ nvidia-docker run -v $(pwd):/root/shared -w "/root/shared" -p 8888:8888 pescapil/deepxde:latest
 
To use a forked version of DeepXDE from inside the Docker, open a Terminal windows and set the ``PYTHONPATH`` adequately using::

$ export PYTHONPATH=$PYTHONPATH:path_to_deepxde