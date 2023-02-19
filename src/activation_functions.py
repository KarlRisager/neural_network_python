import numpy as np


def BinStep_nv(val):
    if val<0:
        return 0
    else:
        return 1
    
def Sigmoid_nv(val):
    return 1/(1+np.exp(-val))

def deriv_Sigmoid_nv(val):
    return Sigmoid_nv(val)(1-Sigmoid_nv(val))

def Tanh_nv(val):
    return (np.exp(val)- np.exp(-val))/(np.exp(val)+ np.exp(-val))

def ReLU_nv(val, leak = 0):
    return max(leak*val, val)

def SoftMax(vals):
    return_array = np.zeros(len(vals))
    s = np.sum(np.exp(vals))
    for i, val in enumerate(vals):
        return_array[i] = np.exp(val)/s
    return return_array

def deriv_SoftMax(vals: np.ndarray):
    '''Currently only returns 1
    '''
    return np.full(vals.size,1)

BinStep = np.vectorize(BinStep_nv)
Sigmoid = np.vectorize(Sigmoid_nv)
deriv_Sigmoid = np.vectorize(deriv_Sigmoid_nv)
Tanh = np.vectorize(Tanh_nv)
ReLU = np.vectorize(ReLU_nv)
