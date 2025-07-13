"""
A
"""
from numpy import ndarray
import numpy.random as numpyrand
from numpy.random import SeedSequence, Generator, PCG64
from src.utility.metaheuristics.nnred import (
    remove_layer, remove_neuron, squeeze_weights, get_capacity
)


def get_neighbors(
    weights: list[ndarray | None], p: float = 0.5,
    seed: int | Generator | None = None
):
    """
    A
    """
    weights = squeeze_weights(weights)  # type: ignore
    if isinstance(seed, Generator):
        gen = seed
    else:
        seeder = SeedSequence()
        if isinstance(seed, int):
            seeder = SeedSequence(seed)
        gen = numpyrand.default_rng(PCG64(seeder))
    capacity = get_capacity(weights)
    neighbors = []
    for layer, cap in enumerate(capacity[1:-1]):
        failed = 0
        for i in range(cap):
            prob = p + (1 - p) * (failed + 1) / cap
            if gen.binomial(1, prob):
                failed = -1
                neighbors += [remove_neuron(weights, layer, [i])]
    for layer in range(len(weights) - 1):
        neighbors += [remove_layer(weights, layer)]
    return neighbors
