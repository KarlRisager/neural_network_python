import numpy as np
from .layer_file import Layer



class NeuralNetwork:

    def __init__(self):
        self.layers = np.array([])
    


    def add_layer(self, width):
        self.layers = np.append(self.layers, Layer(width))

    
    def Forward(self):
        pass

    def BackPropegation(self):
        self
    

    def train(self, data, labels):
        pass

    def show_structur(self):
        print(self.layers)

