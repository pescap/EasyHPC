DeepXDE
=======


To do. Explain the theory for inverse problems, and add references.

Solving a Time-independent PDE: 
-------------------------------
This mathematical example is commonly known, 
but for the original solution in Python you can go to the 
`DeepXDE documentation <https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/helmholtz.2d.dirichlet.html>`_.

Given a wavenumber: :math:`k_{0}=2\pi n` with :math:`n=2`.

You will solve a Helmholtz equation:

.. math:: - u_{xx} - u_{yy} - {k^2}_0u=f,\qquad \Omega={[0,1]}^2

Where :math:`f`:

.. math:: f(x,y)={k^2}_0\sin{(k_0x)}\sin{(k_0y)}

With the Dirichlet boundary conditions:

.. math:: u(x,y)=0,\qquad (x,y) \in \partial \Omega

And finally, the solution is:

.. math:: u(x,y)=\sin{(k_0x)}\sin{(k_0y)}

First, import DeepXDE, Numpy, and TensorFlow (for this example) modules::

    import deepxde as dde
    import numpy as np
    from deepxde.backend import tf

Now, define the general variables of the problem::

    n = 2
    precision_train = 10
    precision_test = 30
    hard_constraint = True
    weights = 100

Defining the epochs (number of steps), the learning rate, the number of layers and nodes, and the activation function::
    
    epochs = 10000
    parameters = [1e-3, 3, 150, "sin"]
        
    sin = tf.sin
    
    learning_rate, num_dense_layers, numb_dense_nodes, activation_func = parameters

Now, we define the PDE residual of the principal equation::

    def pde(x,u):
        dy_xx = dde.grad.hessian(u,x,i=0,j=0)
        dy_yy = dde.grad.hessian(u,x,i=1,j=1)

        f = k0**2*sin(k0*x[:,0:1])*sin(k0*x[:,1:2])
        return -dy_xx-dy_yy-k0**2*u-f

**Note**: The first input is both variables, x and y, because they are the domain. The second argument is the solution :math:`u(x)`.

Now, we introduce the exact solution and the Dirichlet boundary condition::

    def exact_func(x):
        return np.sin(k0*[:,0:1])*np.sin(k0*[:,1:2])
    
    def transformation(x,u):
        res = x[:,0:1]*(1-x[:,0:1])*x[:,1:2]*(1-x[:,1:2])
        return res*u

Now, we define the domain as a rectangle and evaluate the training and test random collocation points. 
This values allow us to obtain collocation points density of 10 (respect 30) points per wavelength along each direction.
For non hard boundary conditions, we must define the boundary and the Dirichlet boundary conditions::

    geom = dde.geometry.Rectangle([0,0],[1,1])
    k0 = 2*np.pi*n
    hx_train = wave_len / precision_train
    nx_train = int(1 / hx_train)

    hx_test = wave_len / precision_test
    nx_test = int(1 / hx_test)

    bc = dde.icbc.DirichletBC(geom, lambda x:0, boundary)

Then, we group all of our data and generate the training and testing points::

    data = dde.data.PDE (
        geom,
        pde,
        bc,
        num_domain = nx_train**2,
        num_boundary = 4*nx_train,
        solution = function,
        num_test = nx_test**2,
    )

Now, it's time to choose the network. In this example, we use a fully connected nerual network of depth 4 (3 hidden layers) and width of 150.
The activation function in this case is `sin` and `Glorot uniform` as initializer::

    network = dde.nn.NN(
        [2] + [numb_dense_nodes]*num_dense_layers + [1], activation, "Glorot uniform"
    )

Now, we have the PDE and the network. With this we build a `Model` and define the optimizer and learning rate::

    model = dde.Model(data, net)

    loss_weights = [1, weights]
    model.compile(
        "adam",
        lr = learning_rate,
        metrics = ["12 relative error"],
        loss_weights = loss_weights
    )

We will train the model for :math:`10000` iterations with Adam optimizer::

    losshistory, train_state = model.train(epochs = epochs)

Saving the plots::

    dde.saveplot(losshistory, train_state, issave = True, isplot = True)

**Full Code**::

    import deepxde as dde
    import numpy as np
    from deepxde.backend import tf

    n = 2
    precision_train = 10
    precision_test = 30
    hard_constraint = True
    weights = 100
    
    epochs = 10000
    parameters = [1e-3, 3, 150, "sin"]
        
    sin = tf.sin
    
    learning_rate, num_dense_layers, numb_dense_nodes, activation_func = parameters

    def pde(x,u):
        dy_xx = dde.grad.hessian(u,x,i=0,j=0)
        dy_yy = dde.grad.hessian(u,x,i=1,j=1)

        f = k0**2*sin(k0*x[:,0:1])*sin(k0*x[:,1:2])
        return -dy_xx-dy_yy-k0**2*u-f

    def exact_func(x):
        return np.sin(k0*[:,0:1])*np.sin(k0*[:,1:2])
    
    def transformation(x,u):
        res = x[:,0:1]*(1-x[:,0:1])*x[:,1:2]*(1-x[:,1:2])
        return res*u

    geom = dde.geometry.Rectangle([0,0],[1,1])
    k0 = 2*np.pi*n
    hx_train = wave_len / precision_train
    nx_train = int(1 / hx_train)

    hx_test = wave_len / precision_test
    nx_test = int(1 / hx_test)

    bc = dde.icbc.DirichletBC(geom, lambda x:0, boundary)

    data = dde.data.PDE (
        geom,
        pde,
        bc,
        num_domain = nx_train**2,
        num_boundary = 4*nx_train,
        solution = function,
        num_test = nx_test**2,
    )

    network = dde.nn.NN(
        [2] + [numb_dense_nodes]*num_dense_layers + [1], activation, "Glorot uniform"
    )

    model = dde.Model(data, net)

    loss_weights = [1, weights]
    model.compile(
        "adam",
        lr = learning_rate,
        metrics = ["12 relative error"],
        loss_weights = loss_weights
    )

    losshistory, train_state = model.train(epochs = epochs)
    dde.saveplot(losshistory, train_state, issave = True, isplot = True)


Bibliography
------------

- DeepXDE: A Deep Learning Library for solving differential equations, Lu, Lu and Meng, Xuhui and Mao, Zhiping and Karniadakis, George Em, SIAM Review (2021) [`link <https://epubs.siam.org/doi/pdf/10.1137/19M1274067>`_]
- Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations, M. Raissi and P. Perdikaris and G.E. Karniadakis, Journal of Computational Physics (2019) [`link <https://www.sciencedirect.com/science/article/pii/S0021999118307125>`_]