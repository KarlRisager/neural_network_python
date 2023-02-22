import numpy as np
from src.activation_functions import *




class NeuralNetwork:

    def __init__(self):
        self.layers = []
        self.nonactivated_layers = []
        self.weights = []
        self.weight_grad = []
        self.biases = []
        self.bias_grad = []
        self.depth = 0
    


    def add_layer(self, width, Input = False, activation=None):
        '''Parameters:\n
            width, The number of nourons in the layer\n
            Activation function is currently global and not local to each layer'''
        #If the layer isn't the input layer we add weights and biases aswell
        #They are initalized randomly. We also add the gradient
        if not(Input):
            previous_layer_length = len(self.layers[-1])
            self.biases.append(np.random.rand(width))
            self.bias_grad.append(np.zeros(width))
            self.weights.append(np.random.rand(width, previous_layer_length))
            self.weight_grad.append(np.zeros((width, previous_layer_length)))
        self.layers.append(np.zeros(width))
        self.nonactivated_layers.append(np.zeros(width))
        self.depth +=1

    
    def forward(self, data, activation=ReLU):
        self.layers[0] = data
        self.nonactivated_layers[0] = data

        for i, layer in enumerate(self.layers[1:]):
            #computing the linear combination
            linear =  self.weights[i].dot(self.layers[i])+ self.biases[i]

            #Updating layers
            self.nonactivated_layers[i+1] = linear
            self.layers[i+1] = activation(linear)

            #using softmax if last layer
            if i==self.depth-2:
                self.layers[i+1] = SoftMax(self.layers[i+1])


    def back_propagation(self, labels: np.ndarray, deriv_act = deriv_ReLU):
        '''Incorrect and unfinnished'''
        #dL/dy
        dY = (self.layers[-1]-labels)
        #dL/dAn
        dAn = dY*deriv_act(self.nonactivated_layers[-1])
        #dL/dWn NOT TRUE
        self.weight_grad[-1] = dAn*self.weights[-1].T
        #dL/dBn
        self.bias_grad[-1] = dAn
        #dZn-1 = Wn^T, dAn-1 = dZn-1*h'(An-2)
        #temp
        dA = dAn
        
        for i in range(2, self.depth):
            dZ = dA @ self.weights[-(i-1)]
            dA = dZ*deriv_act(self.nonactivated_layers[-i])
            self.weight_grad[-i] = dA*self.weights[-i].T
            self.bias_grad[-i] = dA

            





        return
    

    def fit(self, data, labels, num_epochs, step_size):
        '''NOT implemented - Should do loop through epochs. For each iteration do forward propagation on data,
          then do back propagation to calculate gradient and adjust weights and biases acording to gradient and step size\n\n

          data:\n
          \t Numpy 2d array containing training data. Each subarray represents one data point\n

          labels:\n
          \t Numpy array containing the labels of each datapoint\n

          num_epochs:\n
          \t Number of epochs to train the data\n

          step_size:\n
          \t Function determining the stepsize as function of epoch\n
          '''

        raise NotImplementedError('Not implemented yet')

    def one_hot_encode(labels):
        raise NotImplementedError('not yet implemented')

    def show_structur(self):
        print(self.layers)
    
    def loss(self, Y, Y_pred):
        return (1/Y.size) * np.sum(np.power(Y_pred-Y, 2))
