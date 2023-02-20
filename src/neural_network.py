import numpy as np
from src.activation_functions import *




class NeuralNetwork:

    def __init__(self):
        self.layers = []
        self.nonactivated_layers = []
        self.Weights = []
        self.Biases = []
        self.Depth = 0
    


    def add_layer(self, width, Input = False, activation=None):
        '''Parameters:\n
            width, The number of nourons in the layer\n
            Activation function is currently global and not local to each layer'''
        if not(Input):
            self.Biases.append(np.random.rand(width))
            self.Weights.append(np.random.rand(len(self.layers[-1]), width))
        self.layers.append(np.zeros(width))
        self.Depth +=1

    
    def Forward(self, data, activation=ReLU):
        #not used yet
        self.nonactivated_layers = self.layers
        self.layers[0] = data
        self.nonactivated_layers = data
        for i, layer in enumerate(self.layers[1:]):
            self.layers[i+1] = activation(np.dot(self.layers[i], self.Weights[i])+ self.Biases[i])
            if i==self.Depth-2:
                self.layers[i+1] = SoftMax(self.layers[i+1])
            print(self.layers[i+1])


    def BackPropagation(self, Labels: np.ndarray):
        '''Incorrect and unfinnished'''
        m = Labels.size
        dZ2 = 2(self.layers[-1]-Labels)
        dW2 = (1/m)*dZ2.dot(self.layers[-2].T)
        dB2 = (1/m)*np.sum(dZ2)

        dZ1 = self.Weights[-2].T.dot(dZ2)*deriv_ReLU()




        return
    

    def train(self, data, labels, num_epochs):
        pass

    def show_structur(self):
        print(self.layers)
    
    def Loss(self, Y, Y_pred):
        return (1/Y.size) * np.sum(np.power(Y_pred-Y, 2))
