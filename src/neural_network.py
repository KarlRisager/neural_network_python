import numpy as np
from src.activation_functions import *




class NeuralNetwork:

    def __init__(self):
        self.layers = []
        self.Weights = []
        self.Biases = []
    


    def add_layer(self, width, Input = False):
        if not(Input):
            self.Biases.append(np.random.rand(width))
            self.Weights.append(np.random.rand(len(self.layers[-1]), width))
        self.layers.append(np.zeros(width))

    
    def Forward(self, data, activation=SoftMax):
        self.layers[0] = data
        for i, layer in enumerate(self.layers[1:]):
            self.layers[i+1] = activation(np.dot(self.layers[i], self.Weights[i])+ self.Biases[i])
            print(self.layers[i+1])


    def BackPropegation(self, Labels: np.ndarray):
        '''Incorrect and unfinnished'''
        m = Labels.size
        dZ2 = 2(self.layers[-1]-Labels)
        dW2 = (1/m)*dZ2.dot(self.layers[-2].T)*deriv_SoftMax(dZ2)
        dB2 = (1/m)*np.sum(dZ2)

        dZ1 = self.layers[-2].T.dot(dZ2)*deriv_Sigmoid()




        return
    

    def train(self, data, labels):
        pass

    def show_structur(self):
        print(self.layers)

