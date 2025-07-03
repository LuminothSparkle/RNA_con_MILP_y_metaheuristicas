"""
A
"""
import numpy
import numpy.random as numpyrand
from numpy.random import PCG64, SeedSequence


def simulated_annealing(
    black_box_fn: (...), x_0, y_0,
    t_i: float, neighbor_fn: (...),
    iterations: int, t_decay: float,
    seed: int | None = None
):
    """
    A
    """
    ss = SeedSequence()
    if seed is not None:
        ss = SeedSequence(seed)
    gen = numpyrand.default_rng(PCG64(ss))
    x = x_0
    y = y_0
    x_best = x
    y_best = y
    temperature = t_i
    for _ in range(iterations):
        neighbor_x = neighbor_fn(x)
        neighbor_y = black_box_fn(neighbor_x)
        if neighbor_y < y_best:
            x_best, y_best = neighbor_x, neighbor_y
        if bool(gen.binomial(
            1, min(1, numpy.exp(-(neighbor_y - y) / temperature))
        )):
            x, y = neighbor_x, neighbor_y
        temperature *= t_decay
    return x_best, y_best
