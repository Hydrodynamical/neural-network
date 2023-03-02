###REQUIRED LIBRARIES###
import math 
import numpy as np
########################

###GPU SPEEDUP LIBRARY###
#CuPy: cupy.dev
#import cupy as cp
#########################

####USEFUL CONSTANTS####
pii = 1/math.pi   
########################

class DenseNetwork:
    """This class defines a dense neural network"""
    def __init__(self, arch, sigma = 'arctan', d_sigma = 'd_arctan', rand_scale = 1):
        #TODO: ADD ReLu, softmax, etc
        if sigma == 'arctan':
            def s(x):
                return np.arctan(x)*pii + 0.5
            self.sigma = s
        else:
            self.sigma = sigma
        if d_sigma == 'd_arctan':
            def d_s(x):
                return pii/(1+ (x ** 2)) 
            self.d_sigma = d_s
        else:
            self.d_sigma = d_sigma
        self.arch = arch #network architecture as list
        self.rand_scale = rand_scale #scale of randomness 
        self.depth = len(arch) #number of layers of network
        
        #Build network weights, biases, activations, and z, along with gradients
        self.weight = []
        self.bias = []
        self.activation = []
        self.z = []
        self.grad_weight = []
        self.grad_bias = []
        self.grad_activation = []
        for n in range(1,self.depth):
            self.weight.append(rand_scale*np.random.randn(self.arch[n],self.arch[n-1]))
            self.bias.append(rand_scale*np.random.randn(self.arch[n]))
            self.activation.append(np.zeros([self.arch[n]]))
            self.z.append(np.zeros([self.arch[n]]))
            self.grad_weight.append(np.zeros([self.arch[n],self.arch[n-1]]))
            self.grad_bias.append(np.zeros([self.arch[n]]))
            self.grad_activation.append(np.zeros([self.arch[n]]))

    def evaluate(network, input_data):
        """Evaluate the network on the input_data"""
        #Check if input is allowed
        if network.arch[0] != len(input_data):
            raise TypeError(f"ERROR: Input dimensions and network input layer dimension don't match.")
        network.z[0] = np.matmul(network.weight[0],input_data) + network.bias[0]
        network.activation[0] = network.sigma(network.z[0])
        for n in range(1,network.depth-1):
            network.z[n] = np.matmul(network.weight[n],network.activation[n-1])+network.bias[n]
            network.activation[n] = network.sigma(network.z[n])
        return network.activation[-1]
        
    def gradient(network, input_data, expected_output, loss ='L2'):
        """Compute the gradient of the network
        Store in network.grad_*
        """
        if network.arch[-1] != len(expected_output):
            raise TypeError("Expected output dimension and network output layer dimension don't match")
        network.grad_activation[network.depth-2] = network.evaluate(input_data)-expected_output
        for n in range(network.depth-2,0,-1):
            network.grad_bias[n] = (network.d_sigma(network.z[n]))*(network.grad_activation[n])
            network.grad_weight[n] = np.tensordot(network.grad_bias[n],network.activation[n-1],axes = 0)
            network.grad_activation[n-1] = np.matmul(np.transpose(network.grad_bias[n]),network.weight[n])
        network.grad_bias[0] = network.d_sigma(network.z[0])*network.grad_activation[0]
        network.grad_weight[0] = np.tensordot(network.grad_bias[0],input_data,axes = 0)
        return

    def do_GD(network, step_size):
        """Update the network via gradient descent (GD)"""
        for n in range(network.depth-1):
            network.weight[n] = network.weight[n] - step_size*network.grad_weight[n]
            network.bias[n] = network.bias[n] - step_size*network.grad_bias[n]
        return
    def print_state(self):
        print(f"Network weight: \n{self.weight}")
        print(f"Network bias: \n{self.bias}")
        print(f"Network z: \n{self.z}")
        print(f"Network gradient of weight: \n{self.grad_weight}")
        print(f"Network gradient of bias: \n{self.grad_bias}")
        print(f"Network gradient of activation: \n{self.grad_activation}")
        print(f"Network sigma: \n{self.sigma}")
        print(f"Network d_sigma: \n{self.d_sigma}")
        return
    
network = DenseNetwork([1,2,3])
network.print_state()
input_data = np.array([1])
expected_output = np.array([1,0,1,1])
network.evaluate(input_data)
network.gradient(input_data, expected_output)
network.print_state()
network.do_GD(0.001)
network.print_state()
for k in range(1000):
    network.gradient(input_data, expected_output)
    network.do_GD(1)
print(network.evaluate(input_data))