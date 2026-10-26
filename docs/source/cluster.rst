=====================
Working on a cluster
=====================

A cluster is a group of machines (nodes) shared by many users. You connect to a login node, prepare your code and data there, and ask a scheduler to run your jobs on the compute nodes, which have the GPUs. Most academic clusters use the `Slurm <https://slurm.schedmd.com>`__ scheduler.

The host names, partitions and paths below are placeholders. Ask the administrators of your cluster for the actual values.

.. warning::

    Never train on the login node. It is shared by all the users to edit files and submit jobs. Run computations through the scheduler.


Connect with SSH
----------------

Create an SSH key and send it to the cluster, so you do not type your password at each connection::

    $ ssh-keygen -t ed25519
    $ ssh-copy-id user@login.cluster.example.org

Save the connection settings in ``~/.ssh/config`` on your computer:

.. code-block:: text

    Host cluster
        HostName login.cluster.example.org
        User user

    Host gpu-node
        HostName gpu-node.internal
        User user
        ProxyJump cluster

Now ``ssh cluster`` connects to the login node, and ``ssh gpu-node`` reaches a node that is only visible from inside the cluster, through the login node.


Keep a session alive with tmux
******************************

A program started in an SSH session stops when the connection drops. ``tmux`` keeps a session running on the server:

::

    $ tmux new -s work          # Start a session called work
    $ tmux attach -t work       # Reconnect to it later
    $ tmux ls                   # List your sessions

Inside tmux, press ``Ctrl-b`` then ``d`` to detach and leave it running.


Move files
----------

``rsync`` copies only what changed, and can resume an interrupted transfer::

    $ rsync -avP my_project/ cluster:my_project/
    $ rsync -avP cluster:my_project/results/ results/

The trailing ``/`` matters: ``my_project/`` copies the content of the directory, ``my_project`` copies the directory itself. ``scp`` works for single files::

    $ scp data.csv cluster:


Storage
-------

Clusters usually have several file systems:

- The home directory is small, has a quota and is often backed up. Keep your code and configuration there.
- A scratch or work directory is large and fast, but is not backed up and may be cleaned periodically. Keep datasets, checkpoints and results there.
- Each compute node may have a local disk (often ``/tmp``), which is the fastest storage but is erased at the end of the job.

Check your usage before you hit the quota::

    $ df -h ~              # Free space on the file system of your home
    $ du -sh ~/* | sort -h # Size of each directory in your home

Caches of Python packages and datasets fill the home directory quickly. Point them to the scratch directory, for example::

    $ export PIP_CACHE_DIR=/scratch/user/.cache/pip
    $ export HF_HOME=/scratch/user/.cache/huggingface


Software environments
---------------------

Many clusters provide software through environment modules::

    $ module avail              # List the available software
    $ module load cuda          # Load a module
    $ module list               # Show the loaded modules

For Python packages, create an environment of your own, with conda (see :doc:`sysadmin`) or with ``venv``::

    $ python -m venv ~/envs/my_env
    $ source ~/envs/my_env/bin/activate
    $ pip install torch numpy


Containers with Apptainer
*************************

Clusters rarely allow Docker, because it needs administrator rights. `Apptainer <https://apptainer.org/docs/user/latest/>`__ (formerly Singularity) runs the same images as a regular user. Convert a Docker image into an Apptainer image, then run a command inside it with access to the GPUs (``--nv``)::

    $ apptainer pull deepxde.sif docker://pescapil/deepxde:latest
    $ apptainer exec --nv deepxde.sif python my_script.py

Your home directory and the current directory are available inside the container.


Slurm
-----

Look at the cluster and the queue
*********************************

::

    $ sinfo                     # Partitions (groups of nodes) and their state
    $ squeue -u $USER           # Your pending and running jobs
    $ scancel JOBID             # Cancel a job
    $ sacct -j JOBID            # State, duration and exit code of a finished job


Interactive jobs
****************

To test your code on a compute node, ask for an interactive shell with one GPU for one hour::

    $ srun --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=01:00:00 --pty bash

When the job starts, your prompt moves to the compute node. Type ``exit`` to release it.


Batch jobs
**********

For long runs, write a job script, for example ``train.sh``:

.. code-block:: bash

    #!/bin/bash
    #SBATCH --job-name=train
    #SBATCH --partition=gpu
    #SBATCH --gres=gpu:1
    #SBATCH --cpus-per-task=4
    #SBATCH --mem=16G
    #SBATCH --time=12:00:00
    #SBATCH --output=%x-%j.out

    source ~/envs/my_env/bin/activate
    nvidia-smi
    python train.py --epochs 100

The ``#SBATCH`` lines are read by Slurm and request the resources. ``%x`` and ``%j`` in the output file name are replaced by the job name and the job identifier. Submit the script and follow its output::

    $ sbatch train.sh
    $ tail -f train-123456.out

Ask for the resources you need and a realistic time limit. Smaller jobs start sooner, and a job that exceeds its ``--time`` is stopped.


Job arrays
**********

To run the same script with many parameters, for example 10 random seeds, use a job array:

.. code-block:: bash

    #!/bin/bash
    #SBATCH --job-name=seeds
    #SBATCH --partition=gpu
    #SBATCH --gres=gpu:1
    #SBATCH --time=02:00:00
    #SBATCH --array=0-9
    #SBATCH --output=%x-%A_%a.out

    source ~/envs/my_env/bin/activate
    python train.py --seed $SLURM_ARRAY_TASK_ID

Slurm starts 10 jobs, each with a different value of ``SLURM_ARRAY_TASK_ID``. In the output file name, ``%A`` is the identifier of the array and ``%a`` the index of the task.


Jupyter on a compute node
*************************

Start an interactive job, note the name of the node it runs on (for example ``gpu-node``) and start Jupyter there::

    $ jupyter lab --no-browser --port 8888

From your computer, forward the port through the login node::

    $ ssh -L 8888:localhost:8888 -J cluster user@gpu-node

and open the URL printed by Jupyter in your browser.

The `Slurm documentation <https://slurm.schedmd.com/quickstart.html>`__ describes all the commands and options.
