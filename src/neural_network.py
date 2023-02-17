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
            self.Weights.append(np.zeros((len(self.layers[-1]), width)))
        self.layers.append(np.zeros(width))

    
    def Forward(self):
        pass

    def BackPropegation(self):
        self
    

    def train(self, data, labels):
        pass

    def show_structur(self):
        print(self.layers)

