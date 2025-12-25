"""
A
"""
from copy import deepcopy
import numpy
from numpy import ndarray


def remove_neuron(
    weights: list[ndarray | None],
    hidden_layer: int,
    hidden_neurons: list[int]
):
    """
    A
    """
    assert hidden_layer < len(weights) - 1
    if weights[hidden_layer] is None:
        return None
    pre_w = weights[hidden_layer].T  # type: ignore
    next_layer, pos_w = next(
        (hidden_layer + 1 + i, weight.T)
        for i, weight in enumerate(weights[(hidden_layer + 1):])
        if weight is not None
    )
    new_pre_w = pre_w[:, [
        neuron not in hidden_neurons
        for neuron in range(pre_w.shape[1])
    ]]
    pre_wv = numpy.sum(a=pre_w[:, [
        neuron in hidden_neurons
        for neuron in range(pre_w.shape[1])
    ]], axis=1)
    new_pre_w[:, :] += pre_wv.reshape((-1,1)) / new_pre_w.shape[1]
    new_pos_w = pos_w[[
        neuron not in hidden_neurons
        for neuron in range(pos_w.shape[0])
    ], :]
    pos_wv = numpy.sum(a=pos_w[[
        neuron in hidden_neurons
        for neuron in range(pos_w.shape[0])
    ], :], axis=0)
    new_pos_w[:, :] += pos_wv.reshape((1,-1)) / new_pos_w.shape[0]
    new_weights = deepcopy(weights)
    new_weights[hidden_layer] = new_pre_w.T
    new_weights[next_layer] = new_pos_w.T
    return new_weights


def remove_layer(
    weights: list[ndarray | None],
    hidden_layer: int
):
    """
    A
    """
    assert hidden_layer < len(weights) - 1
    if weights[hidden_layer] is None:
        return None
    pre_w = weights[hidden_layer].T  # type: ignore
    next_layer, pos_w = next(
        ((hidden_layer + 1 + i, weight.T)
        for i, weight in enumerate(weights[(hidden_layer + 1):])
        if weight is not None),
        (len(weights) - 1, None)
    )
    if pos_w is None:
        return None
    new_pre_w = numpy.matmul(numpy.matmul(pre_w, numpy.ones((pre_w.shape[1], pos_w.shape[0]))), pos_w)
    new_weights = deepcopy(weights)
    new_weights[hidden_layer] = new_pre_w.T
    new_weights[next_layer] = None
    return new_weights


def squeeze_weights(weights: list[ndarray | None]):
    """
    A
    """
    return [weight for weight in weights if weight is not None]


def get_capacity(weights: list[ndarray | None]):
    """
    A
    """
    return [
        *(
            weight.shape[1] - 1
            if weight is not None
            else None
            for weight in weights
        ),
        next(
            weight.shape[0]
            for weight in reversed(weights)
            if weight is not None
        )
    ]
