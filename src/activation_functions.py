import numpy as np


def BinStep(val):
    if val<0:
        return 0
    else:
        return 1
    
def Sigmoid(val):
    return 1/(1+np.exp(-val))

def Tanh(val):
    return (np.exp(val)- np.exp(-val))/(np.exp(val)+ np.exp(-val))

def ReLU(val, leak = 0):
    return max(leak*val, val)
