from numpy import ndarray
import numpy

def sigmoid(x: ndarray):
    return numpy.tanh(x / 2) / 2 + 1 / 2