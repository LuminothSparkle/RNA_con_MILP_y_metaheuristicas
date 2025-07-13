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
        return weights
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
    pre_wv = pre_w[:, [
        neuron in hidden_neurons
        for neuron in range(pre_w.shape[1])
    ]]
    new_pre_w[:, :] += pre_wv / new_pre_w.shape[1]
    new_pos_w = pos_w[[
        neuron not in hidden_neurons
        for neuron in range(pos_w.shape[0])
    ], :]
    pos_wv = pos_w[[
        neuron in hidden_neurons
        for neuron in range(pos_w.shape[0])
    ], :]
    new_pos_w[:, :] += pos_wv / new_pos_w.shape[0]
    new_weights = deepcopy(weights)
    new_weights[hidden_layer] = new_pre_w.T
    new_weights[next_layer] = new_pos_w.T


def remove_layer(
    weights: list[ndarray | None],
    hidden_layer: int
):
    """
    A
    """
    assert hidden_layer < len(weights) - 1
    hidden_layer = hidden_layer + 1
    if weights[hidden_layer] is None:
        return weights
    pos_w = weights[hidden_layer].T  # type: ignore
    next_layer, pre_w = next(
        (hidden_layer - 1 - i, weight.T)
        for i, weight in enumerate(weights[(hidden_layer - 1)::-1])
        if weight is not None
    )
    new_pre_w = numpy.matmul(pre_w, pos_w)
    new_pos_w = None
    new_weights = deepcopy(weights)
    new_weights[hidden_layer] = new_pos_w
    new_weights[next_layer] = new_pre_w.T
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
        weights[0].shape[0],  # type: ignore
        *(
            weight.shape[1]
            if weight is not None
            else 0
            for weight in weights
        )
    ]
