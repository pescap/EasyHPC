**Theory Behind Inverse Problems** (To do. Explain the theory for inverse problems, and add references.)

**Summary of typical industrial process engineering problems**
------------

As a review of theory and physical concepts, you can find the class readings by Professor María Thomsen Solis from the Adolfo Ibáñez University and the python file made by Artemio Araya. Completion of this teaching material will continue to be developed after the project.

====

**Momentum Conservation**

`M.T Class Lectures <https://drive.google.com/file/d/1BlXg5ymmiAKZ5J5dcu6JQZWL5Yius6QL/view?usp=sharing>`_.

Python File: https://colab.research.google.com/drive/1PhjGNHVIGK0Jek4NyEskEKfb3Ihl9LZy?usp=sharing

====

**Cinematic**

`M.T Class Lectures <https://drive.google.com/file/d/1YhQSburG66JWN0IEQgdkFFf3OaXRe2GT/view?usp=sharing>`_.

Python File: https://colab.research.google.com/drive/1f8_eslVUXN0-8IDR3ASJoHXNIMRTH7oL?usp=sharing

====

**Static**

`M.T Class Lectures <https://drive.google.com/file/d/1QERWlV-Ztj2wzNyQ8vWU8euGz42OO7R6/view?usp=sharing>`_.

Python File: https://colab.research.google.com/drive/1E2HEK9027jVSRXRMAz5m2LPH0_wtlFtE?usp=sharing

====

**Fluyid Dynamics**

`M.T Class Lectures <https://drive.google.com/file/d/1BxTonblF8azjkXSE-xcZeW8Qyyy8xRzP/view?usp=sharing>`_.

Python File: https://colab.research.google.com/drive/1C82mvWWrylb3vxYrxfU0TRjgb6NntUfd?usp=sharing

`M.T Class Lectures <https://drive.google.com/file/d/140Gbbw9qTX1EN4t7PYPgFnUYadKAPk85/view?usp=sharing>`_.

Python File: https://colab.research.google.com/drive/1ftpYxW6-W1iypHdhIPFn7MFofvb26gNm?usp=sharing


`M.T Class Lectures <https://drive.google.com/file/d/1lpRTIV1evP8OF2US6XV1EnTfpjCHtbeZ/view?usp=sharing>`_.

Python File: https://colab.research.google.com/drive/1bC4P4eJ6GHxDjwqHh73IT9ubplW9VA99?usp=sharing

====

**Heat Transfer**

`M.T Class Lectures <https://drive.google.com/file/d/1YGLTkY-rtHdX8B21JP-L_OlvebzcE9zY/view?usp=sharing>`_.

`M.T Class Lectures <https://drive.google.com/file/d/1W-3-1duyDI8AZVkgq9e_KdDxvRN4ptUU/view?usp=sharing>`_.

Python File: https://colab.research.google.com/drive/1q7A5HT_-1zoukSDZyjal9FPLG_imlm6J?usp=sharing

Python File: https://colab.research.google.com/drive/1JUzUBPEuqnKYCxo6BxNY5sAM3XhbzxeH?usp=sharing

Pyhton File: https://colab.research.google.com/drive/12tM7aDL9_stv7w-v2CeNoZcpgN9NFBf5?usp=sharing

**Heat Transfer Equation**

`M.T Class Lectures <https://drive.google.com/file/d/1zayz8u5zzlt4zTrH9YxvT7nx6CgPJBKC/view?usp=sharing>`_.

Python File: https://colab.research.google.com/drive/1-wCq3TP-9sM7eR7pZ8GUVish4iuIQRMw?usp=sharing

====

**Internal convection**

`M.T Class Lectures <https://drive.google.com/file/d/1NvxHsg0PqwW3cjcR9sYD_jrlve6qPd-G/view?usp=sharing>`_.

Python File: https://colab.research.google.com/drive/16DrJ7VjDnYzaS0USNcz8upAiVFYQG5HF?usp=sharing

====

**External, Internal and Natural Forced Convection**

`M.T Class Lectures <>`_.

Python File: https://colab.research.google.com/drive/19Zq49drvMXavEuzHfWjyTLdhy7crHzZs?usp=sharing

====

**External Convection**

`M.T Class Lectures <https://drive.google.com/file/d/1L1gyne2TV_EMGuxnGup-AIyEHFRVVNPf/view?usp=sharing>`_.

Python File: https://colab.research.google.com/drive/1IAfHsnjMZhQe0GETiYm_dv6Kplqkph5e?usp=sharing


====

**Electromagnetism**

Python File: https://colab.research.google.com/drive/1Fb-Uq1kwUgKcxmRui3k7GBmjz3833q1u?usp=sharing

Python File: https://colab.research.google.com/drive/1hafIuEqEhiioZpQajJWUt6aSsPjIFx2V?usp=sharing


====

**Neural Networks Intuitive Approach**
------------
In simple terms, a neural network is a function with the particular ability to learn to predict complex patterns using
data.

.. image:: ~Image/NeuralNetworkbyAndrewNg.png
(by Andrew Ng, Machine Learning Coursera)

As shown in the figure, this neural network have an input pattern vector s of 3 dimensions. And a output predicted response of 1 dimension.

- The layer 1 have 0 neurons.
- The layer 2 have 3 neurons.
- The layer 3 have 2 neurons.
- The layer 4 have 1 neuron.

For any layer the neurons takes the information of the prevoius layers as a lineal combination of weight and basis parameters, and apply a nonlineal
transformation. Typical nonlineal transformations can be ReLU, Sigmoid, Tanh, etc...

There exist many other activations functions. The structure and other propieties of neural networks are goning to be discussed in Mathematical Statistics Section.

In this project, we will use neural networks with only 2 input patern vector that can be the following:

- time and a spatial variable
- two spatial variables

And a output pattern vector that can be:

- a vector belonging to a vector field (For example: Electric Field, Magnetic Field, heat flux, Force field)
- scalar value belonging to a scalar field (Tempeature, mass density)


To optimize the model we need data. Different points in space-time or space-space associated vectors or scalar value belonging to our vector or scalar field.

First, we will use the maximum likelihood method to define the optimization problem, which under a series of assumptions consists of find the parameters of the network that minimizes a type of mean square error (the loss function) between the predictions and the values observed in the database. This type of cost function (loss function) commonly originates when we assume that the values that we want to predict in the database have a random error that distributes normally with constant covariance matrix, and that these random error can also be related.

Second, we will use the gradient descent algorithm to find the network parameters that best fit our predictions.

To complete with the previous task, we will use the backpropagation algorithm to calculate in each "epoch" the evaluation in the gradient function of the cost function for a particular network parameters (A concatenation of arrays of real numbers).

Then with the gradient descent it varies its parameters until it reaches the optimal solution.

These videos explain in a more intuitive way how neural networks work.

What is a neural network?
https://www.youtube.com/watch?v=n1l-9lIMW7E&list=PLpFsSf5Dm-pd5d3rjNtIXUHT-v7bdaEIe&index=2

Supervised Learning with a Neural Network
https://www.youtube.com/watch?v=BYGpKPY9pO0

Mathematic Details About Deep Neural Networks applied to Physics
https://www.overleaf.com/5389572137znnjcpqctqxj




**Mathematical statistics Approach**
------------


**Classic Optimization Methods: Gradient Descent & Backpropagation**
------------



Bibliography
------------

