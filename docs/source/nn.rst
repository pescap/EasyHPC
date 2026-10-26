=====================
Neural Network Theory
=====================

This page explains what a neural network computes, why it is trained by minimizing a mean squared error, and how physics-informed neural networks (PINNs) use the same ideas to solve differential equations.


Neural networks
---------------

In simple terms, a neural network is a function with parameters that can be tuned to fit data.

A fully connected network with :math:`L` layers takes an input vector :math:`x` and computes

.. math::

    a^{(0)} = x, \qquad
    a^{(l)} = \sigma\left( W^{(l)} a^{(l-1)} + b^{(l)} \right), \quad l = 1, \dots, L-1, \qquad
    h_\theta(x) = W^{(L)} a^{(L-1)} + b^{(L)}.

- :math:`W^{(l)}` is the weight matrix and :math:`b^{(l)}` the bias vector of layer :math:`l`. Together they form the parameters :math:`\theta` of the network.
- :math:`\sigma` is a nonlinear function applied to each entry, called the activation function. Common choices are ReLU, the sigmoid, :math:`\tanh` and :math:`\sin`.
- The number of rows of :math:`W^{(l)}` is the number of neurons of layer :math:`l`.

For example, a network with 3 inputs, two hidden layers of 3 and 2 neurons, and 1 output computes

.. math::

    h_\theta(x) = W^{(3)} \sigma\left( W^{(2)} \sigma\left( W^{(1)} x + b^{(1)} \right) + b^{(2)} \right) + b^{(3)},

with :math:`W^{(1)} \in \mathbb{R}^{3 \times 3}`, :math:`W^{(2)} \in \mathbb{R}^{2 \times 3}` and :math:`W^{(3)} \in \mathbb{R}^{1 \times 2}`. For regression, the last layer has no activation, as above.

In the physics problems of this guide, the input is a point in space or in space-time, for example :math:`(x, y)` or :math:`(t, x)`. The output is either

- a scalar field, such as a temperature or a mass density, or
- a vector field, such as a velocity, a heat flux or an electric field.

These videos by Andrew Ng give an intuitive introduction:

- `What is a neural network? <https://www.youtube.com/watch?v=n1l-9lIMW7E>`__
- `Supervised learning with a neural network <https://www.youtube.com/watch?v=BYGpKPY9pO0>`__


Training: why a mean squared error
----------------------------------

Suppose we measure a field :math:`u` with :math:`K` components at points :math:`x_1, \dots, x_m`, and that each measurement has an instrument error:

.. math::

    y_i = u(x_i) + \varepsilon_i, \qquad \varepsilon_i \sim \mathcal{N}(0, s^2 I), \quad i = 1, \dots, m,

with independent errors. We use the network to approximate :math:`u`. The likelihood of the parameters :math:`\theta` given the measurements is

.. math::

    \mathcal{L}(\theta) = \prod_{i=1}^{m} \frac{1}{(2 \pi s^2)^{K/2}} \exp\left( -\frac{\| y_i - h_\theta(x_i) \|^2}{2 s^2} \right),

so that

.. math::

    -\log \mathcal{L}(\theta) = \frac{1}{2 s^2} \sum_{i=1}^{m} \| y_i - h_\theta(x_i) \|^2 + \text{constant}.

Maximizing the likelihood is therefore the same as minimizing the mean squared error (MSE)

.. math::

    J(\theta) = \frac{1}{m} \sum_{i=1}^{m} \| y_i - h_\theta(x_i) \|^2.

This is why MSE is the default loss of most regression problems. In DeepXDE, ``Model.compile`` uses ``loss="MSE"`` by default.

The loss is minimized by gradient descent. Starting from random parameters, each step moves them against the gradient:

.. math::

    \theta_{k+1} = \theta_k - \eta \, \nabla_\theta J(\theta_k),

where :math:`\eta > 0` is the learning rate. The gradient is computed by backpropagation, that is, by applying the chain rule layer by layer. In practice, variants such as Adam are used, and each step uses a random subset of the data (a mini-batch). One pass over the whole dataset is called an epoch.


Physics-informed neural networks
--------------------------------

A PINN approximates the solution :math:`u` of a differential equation with a network :math:`h_\theta`. Consider a problem

.. math::

    \mathcal{N}[u] = f \ \text{in } \Omega, \qquad \mathcal{B}[u] = g \ \text{on } \partial\Omega,

where :math:`\mathcal{N}` is a differential operator and :math:`\mathcal{B}` gives the boundary condition. The network is trained to minimize

.. math::

    J(\theta) = \frac{1}{m_r} \sum_{i=1}^{m_r} \left| \mathcal{N}[h_\theta](x_i) - f(x_i) \right|^2
    + \frac{\lambda}{m_b} \sum_{j=1}^{m_b} \left| \mathcal{B}[h_\theta](z_j) - g(z_j) \right|^2,

where the :math:`x_i` are collocation points inside :math:`\Omega`, the :math:`z_j` points on its boundary and :math:`\lambda > 0` a weight. The derivatives of the network in :math:`\mathcal{N}` are computed exactly by automatic differentiation. When measurements are available, their MSE is added to the loss. :doc:`deepxde` solves a Helmholtz equation this way.


Inverse problems
****************

In an inverse problem, some coefficients of the equation are unknown, for example a diffusivity or a wave speed, and measurements of the solution are available. The unknown coefficients become trainable parameters alongside :math:`\theta`, and the loss includes the misfit with the measurements. Minimizing it recovers both the solution and the coefficients. In DeepXDE, an unknown coefficient is declared with ``dde.Variable`` and passed to ``Model.compile`` through ``external_trainable_variables``.


PINNs and the finite element method
***********************************

The finite element method (FEM) is the standard tool to solve differential equations in engineering, for problems such as heat transfer, fluid flow and electromagnetism. It divides the domain into a mesh, approximates the solution with a piecewise polynomial on each element, and computes the coefficients by solving a sparse linear system. FEM is mature, comes with error estimates, and is fast for forward problems in two and three dimensions.

PINNs approximate the solution with a neural network instead. They need no mesh, only scattered collocation points. Their loss is nonlinear and nonconvex in the parameters and is minimized by gradient descent, which gives no guarantee of reaching the best network. Their strengths are that forward and inverse problems are written with the same code, that measurements are easy to add to the loss, and that they extend to high-dimensional problems where a mesh is not affordable. For standard forward problems in low dimension, PINNs are usually slower and less accurate than FEM (see Grossmann et al., 2024).


Course material
---------------

The engineering problems below come from the course of Professor María Thomsen Solís at Universidad Adolfo Ibáñez. Artemio Araya wrote a Python notebook for each topic, on Google Colab.

- Momentum conservation: `notebook <https://colab.research.google.com/drive/1PhjGNHVIGK0Jek4NyEskEKfb3Ihl9LZy?usp=sharing>`__
- Kinematics: `notebook <https://colab.research.google.com/drive/1f8_eslVUXN0-8IDR3ASJoHXNIMRTH7oL?usp=sharing>`__
- Statics: `notebook <https://colab.research.google.com/drive/1E2HEK9027jVSRXRMAz5m2LPH0_wtlFtE?usp=sharing>`__
- Fluid dynamics: `notebook 1 <https://colab.research.google.com/drive/1C82mvWWrylb3vxYrxfU0TRjgb6NntUfd?usp=sharing>`__, `notebook 2 <https://colab.research.google.com/drive/1ftpYxW6-W1iypHdhIPFn7MFofvb26gNm?usp=sharing>`__, `notebook 3 <https://colab.research.google.com/drive/1bC4P4eJ6GHxDjwqHh73IT9ubplW9VA99?usp=sharing>`__
- Heat transfer: `notebook 1 <https://colab.research.google.com/drive/1q7A5HT_-1zoukSDZyjal9FPLG_imlm6J?usp=sharing>`__, `notebook 2 <https://colab.research.google.com/drive/1JUzUBPEuqnKYCxo6BxNY5sAM3XhbzxeH?usp=sharing>`__, `notebook 3 <https://colab.research.google.com/drive/12tM7aDL9_stv7w-v2CeNoZcpgN9NFBf5?usp=sharing>`__
- Heat equation: `notebook <https://colab.research.google.com/drive/1-wCq3TP-9sM7eR7pZ8GUVish4iuIQRMw?usp=sharing>`__
- Internal convection: `notebook <https://colab.research.google.com/drive/16DrJ7VjDnYzaS0USNcz8upAiVFYQG5HF?usp=sharing>`__
- External, internal and natural convection: `notebook <https://colab.research.google.com/drive/19Zq49drvMXavEuzHfWjyTLdhy7crHzZs?usp=sharing>`__
- External convection: `notebook <https://colab.research.google.com/drive/1IAfHsnjMZhQe0GETiYm_dv6Kplqkph5e?usp=sharing>`__
- Electromagnetism: `notebook 1 <https://colab.research.google.com/drive/1Fb-Uq1kwUgKcxmRui3k7GBmjz3833q1u?usp=sharing>`__, `notebook 2 <https://colab.research.google.com/drive/1hafIuEqEhiioZpQajJWUt6aSsPjIFx2V?usp=sharing>`__


Bibliography
------------

- I. Goodfellow, Y. Bengio and A. Courville. *Deep Learning*. MIT Press, 2016. `deeplearningbook.org <https://www.deeplearningbook.org>`__
- M. Raissi, P. Perdikaris and G. E. Karniadakis. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378:686–707, 2019. `doi:10.1016/j.jcp.2018.10.045 <https://doi.org/10.1016/j.jcp.2018.10.045>`__
- L. Lu, X. Meng, Z. Mao and G. E. Karniadakis. DeepXDE: A deep learning library for solving differential equations. *SIAM Review*, 63(1):208–228, 2021. `doi:10.1137/19M1274067 <https://doi.org/10.1137/19M1274067>`__
- T. G. Grossmann, U. J. Komorowska, J. Latz and C.-B. Schönlieb. Can physics-informed neural networks beat the finite element method? *IMA Journal of Applied Mathematics*, 89(1):143–174, 2024. `doi:10.1093/imamat/hxae011 <https://doi.org/10.1093/imamat/hxae011>`__
