System Administration
======================

Linux command line for beginners
--------------------------------

Here goes a short list of useful commands:

File commands
*******************

$ ls # Show current directory
$ cd # Go to the root directory
$ cd folder_example # Change directory to 'folder_example'
$ cd .. # Go back to the previous directory
$ mkdir folder_example # Create a new folder named 'folder_example'
$ rm file_name  # Delete file named 'file_name'
$ rm -rf folder_example # Delete non empty folder named <folder_example>
$ pwd # Show the current path
$ cat file_name # Show the content of file named 'file_name'

System info
*****************

$ htop # Check memory usage and processes
$ df # Show disk usage
$ date # Show the current date and time

Cluster and Jupyter Notebook
----------------------------

Connect via ssh to ``ip_address`` with username ``user``: ::

    $ ssh user@ip_address

Connect via ssh to the server, enabling port forwarding with port ``8888`` using: ::

    $ ssh -L 8888:localhost:8888 user@ip_address

Once logged into the server, simply run: ::

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

Anaconda
--------
Conda is a package and environment management system, allowing to easily find and install packages. It is very useful to work with different Python packages/libraries versions, and to avoid compatibility issues.

1. Installing Anaconda.
    Unix:
       1. Download the latest version of `Conda <https://www.anaconda.com/products/individual>`_ (copy the link of download)::

            $ curl -O donwload_link

       2. Run::

            $ bash <name of file downloaded>.sh

       3. Follow the instructions on the installer screen.
       4. **Remember** to type **yes**, at the end of everything, this will add **Conda** to **PATH**.
       5. Restart your terminal window.

    MacOS:
       1. Download the latest version of `Conda <https://www.anaconda.com/products/individual>`_ (copy the link of download)::
            
            $ curl -O donwload_link
            
       2. Run::

            $ bash <name of file downloaded>.sh

       3. Follow the instructions on the installer screen.
       4. **Remember** to type **yes**, at the end of everything, this will add **Conda** to **PATH**.
       5. Restart the terminal window.

    **Note**: If conda command doesn't work (skipping step 4), run::
        
        $ conda init

2. Creating/Managing your environments.
    1. To create an environment with a specific version::
        
        $ conda create -n new_env python=<version_number>

    2. Checking the list of your environments::
        
        $ conda env list

    3. Activating/Deactivating an environment::
        
        $ conda activate/deactivate new_env

    4. Install a package::
    
        $ conda install name_of_the_package
        or
        $ pip install name_of_the_package

- Remember that any packages/libraries installed on a specific conda environment are retained there, environments do not share installed packages.
- If you want to know more about Anaconda, you can go to their official documentation `website <https://docs.conda.io/projects/conda/en/latest/index.html>`_.

Docker
------
 
To run a DeepXDE container, run: ::

    $ nvidia-docker run -v $(pwd):/root/shared -w "/root/shared" -p 8888:8888 pescapil/deepxde:latest
 
To use a forked version of DeepXDE from inside the Docker, open a Terminal windows and set the ``PYTHONPATH`` adequately using::

    $ export PYTHONPATH=$PYTHONPATH:path_to_deepxde
