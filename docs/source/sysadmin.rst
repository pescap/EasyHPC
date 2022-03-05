System Adminisitration
======================

Linux command line for beginners
--------------------------------

Here goes a short list of useful commands::

$ ls # Show current directory
$ cd # Go to the root directory
$ pwd # Show the current path


Cluster and Jupyter Notebook
----------------------------

Connect to ssh to the server, enabling port forwarding with port ``8888`` using::

$ ssh -L 8888:localhost:8888 your_username@server_ip_address

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
 
To run a the DeepXDE container, run: ::

$ nvidia-docker run -v $(pwd):/root/shared -w "/root/shared" -p 8888:8888 pescapil/deepxde:latest
