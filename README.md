# Introduction

This project is devoted to making a neural network architecture with minimal library requirements in Python. For CPU only support, only the math package and NumPy will be used. For GPU support, I plan to use CuPy instead of NumPy (this requires a CUDA supported GPU).

See the PDF labeled "Neural Networks: Overview" for the basics of neural networks, from a mathematical perspective.

# DenseNetwork Class
This class contains all dense neural network architectures. Each object in this class has the following ***attributes***:

0. **init** 
   -Initialization of a network has manditory input arch, which is a list of integers. Optional argument are sigma = 'arctan', d_sigma = 'd_arctan',  and rand_scale = 1. 
1. **arch**
   - A list of numbers describing the number of neurons per layer.
2. **weight**
   - A list of the weight matrices per layer.
3. **bias**
   - A list of the biase vectors per layer.
4. **sigma**
   - The non-linearity of the network. sigma = 'arctan' is default.
5. **d_sigma**
   - The derivative of sigma. d_sigma = 'd_arctan' is default.
6. **z**
   - The "response" of the network before applying sigma. This is useful to keep track of when we are computing the activation of the network.
7. **grad_activation**
   - The gradient of the loss with respect to the activations using the L2 loss. This is useful to keep track of when computing the gradient of the network. 
8. **grad_weight**
   - The gradient of the loss with respect to the weights using the L2 loss.
9. **grad_bias** 
   - The gradient of the loss with respect to the biases using the L2 loss.

The ***methods*** for this class are the following:
1. **evaluate**
   - Evaluate network at a given input. TypeError is raised if dimension mismatch. Subfunction of gradient function below.
2. **gradient**
   - Update the gradients of the network given an input data and an expected output using the L2 loss.
3. **do_GD**
   - Update the weights and biases of the network using gradient decent, and a given step size. 
4. **print_state**
   - This is for debugging. Prints out the attributes of the network.
   
# Example Usage
In any Python IDE, you can use the following commands to use the DenseNetwork class:

```python
!git clone ...
import numpy as np
import math
from neuralnetwork.NetworkArch import DenseNetwork
network = DenseNetwork(arch = [10, 5, 1])
input = np.random.randn(10)
network.evaluate(input)
network.print_state()
```
