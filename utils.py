import numpy as np

def sigmoid(z):
    """
    Compute the sigmoid of z

    Parameters
    ----------
    z : numpy array or scalar

    Returns
    -------
    sigmoid of z
    """
    return 1/(1+np.exp(-z))

def sigmoid_backward(dA, z):
    s = 1 / (1 + np.exp(-z))
    return dA * s * (1 - s)


def relu(x):
    """
    Compute the relu of x

    Parameters:
    x -- A scalar or numpy array of any size.

    Return:
    s -- relu(x)
    """
    s = np.maximum(0,x)
    
    return s

def relu_backward(dA, z):
    dz = np.array(dA, copy=True)
    dz[z <= 0] = 0
    return dz

def tanh(z):
    return np.tanh(z)

def tanh_backward(dA, z):
    t = np.tanh(z)
    return dA * (1 - t**2)

def softmax(z):
    shift = z - np.max(z, axis=0, keepdims=True)
    exp_scores = np.exp(shift)
    probs = exp_scores / np.sum(exp_scores, axis=0, keepdims=True)
    return probs

def binary_cross_entropy(Y_hat, Y):
    
    m = Y.shape[1]
    return -1/m * np.sum(Y * np.log(Y_hat) + (1-Y) * np.log(1-Y_hat))

def multiclass_cross_entropy(Y_hat, Y):
    
    m = Y.shape[1]
    return -1/m * np.sum(Y * np.log(Y_hat))