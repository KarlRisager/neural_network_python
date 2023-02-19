import numpy as np
from .layer_file import Layer



class NeuralNetwork:

    def __init__(self):
        self.layers = []
        self.Weights = []
        self.Biases = []
    


    def add_layer(self, width, Input = False):
        if not(Input):
            self.Biases.append(np.zeros(width))
            self.Weights.append(np.random.rand(len(self.layers[-1]), width))
        self.layers.append(np.zeros(width))

    
    def Forward(self, data):
        self.layers[0] = data
        for i, layer in enumerate(self.layers[1:]):
            self.layers[i+1] = np.dot(self.layers[i], self.Weights[i])+ self.Biases[i]
            print(self.layers[i+1])


    def BackPropegation(self):
        self
    

    def train(self, data, labels):
        pass

    def show_structur(self):
        print(self.layers)

