===========================
Distributed PyTorch (DDP)
===========================

PyTorch trains a model on several GPUs with ``DistributedDataParallel`` (DDP), and launches the processes with ``torchrun``. It replaces :doc:`horovod` for new PyTorch projects.

DDP starts one process per GPU. Each process holds a copy of the model and trains on a different part of each epoch. After each backward pass, the processes average their gradients, so all the copies stay identical.


A complete example
------------------

The script below fits a small network to a function of two variables. It runs on GPUs when they are available, and on the CPU otherwise, so you can test it on your laptop. Save it as ``ddp.py``:

.. code-block:: python

    import os

    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, TensorDataset
    from torch.utils.data.distributed import DistributedSampler


    def main():
        local_rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            dist.init_process_group("nccl")
        else:
            device = torch.device("cpu")
            dist.init_process_group("gloo")
        rank = dist.get_rank()

        torch.manual_seed(0)
        x = torch.rand(10000, 2)
        y = torch.sin(6.0 * x[:, :1]) * torch.cos(6.0 * x[:, 1:])
        dataset = TensorDataset(x, y)
        sampler = DistributedSampler(dataset)
        loader = DataLoader(dataset, batch_size=128, sampler=sampler)

        model = torch.nn.Sequential(
            torch.nn.Linear(2, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, 1),
        ).to(device)
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(20):
            sampler.set_epoch(epoch)
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = torch.nn.functional.mse_loss(model(xb), yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if rank == 0:
                print(f"epoch {epoch}  loss {loss.item():.4f}")

        if rank == 0:
            torch.save(model.module.state_dict(), "model.pt")
        dist.destroy_process_group()


    if __name__ == "__main__":
        main()

The differences with a single-GPU script are:

- ``torchrun`` sets the environment variable ``LOCAL_RANK``, the index of the process on its machine. Each process uses the GPU with that index.
- ``dist.init_process_group`` connects the processes: ``nccl`` for GPUs, ``gloo`` for CPUs.
- ``DistributedSampler`` gives each process a different part of the dataset. ``sampler.set_epoch`` shuffles it differently at each epoch.
- ``DDP`` wraps the model and averages the gradients during ``loss.backward()``.
- Only process ``0`` (``rank == 0``) prints and saves the model, so the processes do not write over each other. ``model.module`` is the original model inside the DDP wrapper.

The batch size of 128 is per process: with 4 GPUs, each optimizer step uses 512 examples.


Run on one machine
------------------

To use 4 GPUs of one machine::

    $ torchrun --standalone --nproc_per_node=4 ddp.py

To test on a laptop without GPUs, start 2 processes on the CPU::

    $ torchrun --standalone --nproc_per_node=2 ddp.py

If it hangs on macOS, give the address explicitly::

    $ torchrun --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29500 ddp.py

To choose which GPUs to use, set ``CUDA_VISIBLE_DEVICES`` (see :doc:`sysadmin`).


Run on several nodes with Slurm
-------------------------------

With Slurm, start one ``torchrun`` per node with ``srun``. The ``torchrun`` instances meet on the first node of the job. For 2 nodes with 4 GPUs each:

.. code-block:: bash

    #!/bin/bash
    #SBATCH --job-name=ddp
    #SBATCH --nodes=2
    #SBATCH --ntasks-per-node=1
    #SBATCH --gres=gpu:4
    #SBATCH --cpus-per-task=16
    #SBATCH --time=04:00:00
    #SBATCH --output=%x-%j.out

    source ~/envs/my_env/bin/activate
    head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

    srun torchrun \
        --nnodes=$SLURM_NNODES \
        --nproc_per_node=4 \
        --rdzv_id=$SLURM_JOB_ID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$head_node:29500 \
        ddp.py

The script does not change: ``torchrun`` sets ``LOCAL_RANK`` and the global rank for each of the 8 processes.

See the `DDP tutorial <https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html>`__ and the `torchrun documentation <https://docs.pytorch.org/docs/stable/elastic/run.html>`__ for more.
