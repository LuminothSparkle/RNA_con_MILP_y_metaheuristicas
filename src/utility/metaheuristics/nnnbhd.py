"""
A
"""
from numpy import ndarray
import numpy.random as numpyrand
from numpy.random import SeedSequence, Generator, PCG64
from utility.metaheuristics.nnred import (
    remove_layer, remove_neuron, get_capacity
)


def get_neighbors(
    weights: list[ndarray | None], p: float = 0.5,
    seed: int | Generator | None = None
):
    """
    A
    """
    if isinstance(seed, Generator):
        gen = seed
    else:
        seeder = SeedSequence()
        if isinstance(seed, int):
            seeder = SeedSequence(seed)
        gen = numpyrand.default_rng(PCG64(seeder))
    neighbors = []
    capacity = get_capacity(weights)
    for layer, cap in enumerate(capacity[1:-1]):
        if weights[layer] is not None and cap is not None and cap > 0:
            failed = 1
            for i in range(cap):
                prob = p + (1 - p) * failed * (i + 1) / cap
                if gen.binomial(1, prob) == 1:
                    failed = 0
                    neighbor = remove_neuron(weights, layer, [i])
                    if neighbor is not None:
                        neighbors += [neighbor]
    for layer, cap in enumerate(capacity[1:-1]):
        neighbor = remove_layer(weights, layer)
        if neighbor is not None:
            neighbors += [neighbor]
    return neighbors
