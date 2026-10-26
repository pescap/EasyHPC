=======
DeepXDE
=======

`DeepXDE <https://deepxde.readthedocs.io>`__ is a Python library for physics-informed neural networks (PINNs). This page installs it and solves a Helmholtz equation, step by step.


Installation
------------

DeepXDE runs on top of a deep learning library, called its backend: TensorFlow, PyTorch, JAX or PaddlePaddle. Work in a conda environment (see :doc:`sysadmin`) to avoid conflicts with other projects. For example, with the PyTorch backend::

    $ conda create -n deepxde python=3.12
    $ conda activate deepxde
    $ pip install deepxde torch

Choose the backend with the environment variable ``DDE_BACKEND`` when you run a script::

    $ DDE_BACKEND=pytorch python my_script.py

To check the installation, start Python and import the library::

    $ DDE_BACKEND=pytorch python -c "import deepxde"

The `installation guide <https://deepxde.readthedocs.io/en/latest/user/installation.html>`__ lists the versions supported for each backend.


Use your own copy of DeepXDE
****************************

To modify DeepXDE itself, clone the repository (or your fork of it, see :doc:`github`) and install it in editable mode, so that Python uses your copy of the code::

    $ git clone https://github.com/lululxvi/deepxde.git
    $ cd deepxde
    $ pip install -e .

The ``examples`` directory of the repository contains many ready-to-run problems.


Solving a time-independent PDE
------------------------------

.. note::

    This example follows `Helmholtz equation over a 2D square domain <https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/helmholtz.2d.dirichlet.html>`__ in the DeepXDE documentation. The full script is `Helmholtz_Dirichlet_2d.py <https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Helmholtz_Dirichlet_2d.py>`__.


Problem
*******

Given the wavenumber :math:`k_0 = 2\pi n` with :math:`n = 2`, we solve the Helmholtz equation

.. math::

    -u_{xx} - u_{yy} - k_0^2 u = f, \qquad (x, y) \in \Omega = [0, 1]^2,

with the source term

.. math::

    f(x, y) = k_0^2 \sin(k_0 x) \sin(k_0 y)

and the Dirichlet boundary condition

.. math::

    u(x, y) = 0, \qquad (x, y) \in \partial \Omega.

The exact solution is

.. math::

    u(x, y) = \sin(k_0 x) \sin(k_0 y).


Step by step
************

Import DeepXDE and NumPy::

    import deepxde as dde
    import numpy as np

Define the parameters of the problem: the number of training and test points per wavelength, whether the boundary condition is imposed exactly (``hard_constraint``), the weight of the boundary loss when it is not, and the number of training iterations::

    n = 2
    precision_train = 10
    precision_test = 30
    hard_constraint = True
    weights = 100
    iterations = 5000

Define the learning rate, the number of hidden layers, the number of neurons per layer and the activation function::

    learning_rate, num_dense_layers, num_dense_nodes, activation = 1e-3, 3, 150, "sin"

Define the residual of the PDE. ``x`` holds the coordinates :math:`(x, y)` of a batch of points and ``u`` the output of the network at those points::

    sin = dde.backend.sin

    def pde(x, u):
        u_xx = dde.grad.hessian(u, x, i=0, j=0)
        u_yy = dde.grad.hessian(u, x, i=1, j=1)
        f = k0**2 * sin(k0 * x[:, 0:1]) * sin(k0 * x[:, 1:2])
        return -u_xx - u_yy - k0**2 * u - f

``dde.backend.sin`` works with any backend. The function uses ``k0``, which is defined below, before the training starts.

Define the exact solution, used to measure the error::

    def func(x):
        return np.sin(k0 * x[:, 0:1]) * np.sin(k0 * x[:, 1:2])

With ``hard_constraint = True``, the boundary condition is built into the network: its output is multiplied by :math:`x(1-x)y(1-y)`, which vanishes on the boundary of the square. Otherwise, the boundary condition is added to the loss::

    def transform(x, u):
        res = x[:, 0:1] * (1 - x[:, 0:1]) * x[:, 1:2] * (1 - x[:, 1:2])
        return res * u

    def boundary(_, on_boundary):
        return on_boundary

Define the domain, the wavenumber and the number of points along each direction. With 10 points per wavelength for training and 30 for testing::

    geom = dde.geometry.Rectangle([0, 0], [1, 1])
    k0 = 2 * np.pi * n
    wave_len = 1 / n

    hx_train = wave_len / precision_train
    nx_train = int(1 / hx_train)

    hx_test = wave_len / precision_test
    nx_test = int(1 / hx_test)

    if hard_constraint:
        bc = []
    else:
        bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary)

Group the PDE, the boundary condition and the training and test points::

    data = dde.data.PDE(
        geom,
        pde,
        bc,
        num_domain=nx_train**2,
        num_boundary=4 * nx_train,
        solution=func,
        num_test=nx_test**2,
    )

Build a fully connected network with 2 inputs, 3 hidden layers of 150 neurons and 1 output, with the ``sin`` activation and the Glorot uniform initializer::

    net = dde.nn.FNN(
        [2] + [num_dense_nodes] * num_dense_layers + [1], activation, "Glorot uniform"
    )

    if hard_constraint:
        net.apply_output_transform(transform)

Build the model and choose the Adam optimizer. The metric is the relative :math:`L^2` error against the exact solution::

    model = dde.Model(data, net)

    if hard_constraint:
        model.compile("adam", lr=learning_rate, metrics=["l2 relative error"])
    else:
        model.compile(
            "adam",
            lr=learning_rate,
            metrics=["l2 relative error"],
            loss_weights=[1, weights],
        )

Train the model and save the loss history and the solution::

    losshistory, train_state = model.train(iterations=iterations)
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)


Full code
*********

::

    import deepxde as dde
    import numpy as np

    n = 2
    precision_train = 10
    precision_test = 30
    hard_constraint = True
    weights = 100
    iterations = 5000

    learning_rate, num_dense_layers, num_dense_nodes, activation = 1e-3, 3, 150, "sin"

    sin = dde.backend.sin


    def pde(x, u):
        u_xx = dde.grad.hessian(u, x, i=0, j=0)
        u_yy = dde.grad.hessian(u, x, i=1, j=1)
        f = k0**2 * sin(k0 * x[:, 0:1]) * sin(k0 * x[:, 1:2])
        return -u_xx - u_yy - k0**2 * u - f


    def func(x):
        return np.sin(k0 * x[:, 0:1]) * np.sin(k0 * x[:, 1:2])


    def transform(x, u):
        res = x[:, 0:1] * (1 - x[:, 0:1]) * x[:, 1:2] * (1 - x[:, 1:2])
        return res * u


    def boundary(_, on_boundary):
        return on_boundary


    geom = dde.geometry.Rectangle([0, 0], [1, 1])
    k0 = 2 * np.pi * n
    wave_len = 1 / n

    hx_train = wave_len / precision_train
    nx_train = int(1 / hx_train)

    hx_test = wave_len / precision_test
    nx_test = int(1 / hx_test)

    if hard_constraint:
        bc = []
    else:
        bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary)

    data = dde.data.PDE(
        geom,
        pde,
        bc,
        num_domain=nx_train**2,
        num_boundary=4 * nx_train,
        solution=func,
        num_test=nx_test**2,
    )

    net = dde.nn.FNN(
        [2] + [num_dense_nodes] * num_dense_layers + [1], activation, "Glorot uniform"
    )

    if hard_constraint:
        net.apply_output_transform(transform)

    model = dde.Model(data, net)

    if hard_constraint:
        model.compile("adam", lr=learning_rate, metrics=["l2 relative error"])
    else:
        model.compile(
            "adam",
            lr=learning_rate,
            metrics=["l2 relative error"],
            loss_weights=[1, weights],
        )

    losshistory, train_state = model.train(iterations=iterations)
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)


Bibliography
------------

- L. Lu, X. Meng, Z. Mao and G. E. Karniadakis. DeepXDE: A deep learning library for solving differential equations. *SIAM Review*, 63(1):208–228, 2021. `doi:10.1137/19M1274067 <https://doi.org/10.1137/19M1274067>`__
- M. Raissi, P. Perdikaris and G. E. Karniadakis. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378:686–707, 2019. `doi:10.1016/j.jcp.2018.10.045 <https://doi.org/10.1016/j.jcp.2018.10.045>`__
