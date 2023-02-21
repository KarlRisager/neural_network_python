import numpy as np
from src.activation_functions import *




class NeuralNetwork:

    def __init__(self):
        self.layers = []
        self.nonactivatedLayers = []
        self.weights = []
        self.weightGrad = []
        self.biases = []
        self.biasGrad = []
        self.Depth = 0
    


    def addLayer(self, width, Input = False, activation=None):
        '''Parameters:\n
            width, The number of nourons in the layer\n
            Activation function is currently global and not local to each layer'''
        #If the layer isn't the input layer we add weights and biases aswell
        #They are initalized randomly. We also add the gradient
        if not(Input):
            previousLayerLength = len(self.layers[-1])
            self.biases.append(np.random.rand(width))
            self.biasGrad.append(np.zeros(width))
            self.weights.append(np.random.rand(width, previousLayerLength))
            self.weightGrad.append(np.zeros((width, previousLayerLength)))
        self.layers.append(np.zeros(width))
        self.nonactivatedLayers.append(np.zeros(width))
        self.Depth +=1

    
    def Forward(self, data, activation=ReLU):
        self.layers[0] = data
        self.nonactivatedLayers[0] = data

        for i, layer in enumerate(self.layers[1:]):
            #computing the linear combination
            linear =  self.weights[i].dot(self.layers[i])+ self.biases[i]

            #Updating layers
            self.nonactivatedLayers[i+1] = linear
            self.layers[i+1] = activation(linear)

            #using softmax if last layer
            if i==self.Depth-2:
                self.layers[i+1] = SoftMax(self.layers[i+1])


    def backPropagation(self, labels: np.ndarray, deriv_act = deriv_ReLU):
        '''Incorrect and unfinnished'''
        m = labels.size
        #dL/dy
        dY = (self.layers[-1]-labels)
        #dL/dAn
        dAn = dY*deriv_act(self.nonactivatedLayers[-1])
        #dL/dWn NOT TRUE
        dWn = dAn*self.weights[-1]
        #dL/dBn
        dBn = dAn
        #dZn-1 = Wn^T, dAn-1 = dZn-1*h'(An-2)
        #temp
        dA = dAn
        
        for i in range(2, self.Depth+1):
            dZ = dA.dot(np.transpose(self.weights[-(i-1)]))#i or (i-1)?
            dA = dZ*deriv_act(self.nonactivatedLayers[-i])
            #NOT TRUE
            dW = dA*self.weights[-i]#NOOOOOO
            dB = dA

            


        raise NotImplementedError('Not implemented yet')
        dZ2 = 2(self.layers[-1]-Labels)
        dW2 = (1/m)*dZ2.dot(self.layers[-2].T)
        dB2 = (1/m)*np.sum(dZ2)

        dZ1 = self.Weights[-2].T.dot(dZ2)*deriv_ReLU()




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

    def showStructur(self):
        print(self.layers)
    
    def loss(self, Y, Y_pred):
        return (1/Y.size) * np.sum(np.power(Y_pred-Y, 2))
