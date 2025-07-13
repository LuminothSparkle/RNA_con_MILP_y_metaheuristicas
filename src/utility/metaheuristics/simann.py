"""
A
"""
import numpy
import numpy.random as numpyrand
from numpy import ndarray
from numpy.random import PCG64, SeedSequence, Generator
from torch.utils.data import Subset
from src.utility.metaheuristics.fitness import WeightFitnessCalculator
from src.utility.metaheuristics.nnnbhd import get_neighbors


def simulated_annealing(
    t_i: float,
    iterations: int, t_decay: float,
    weights: list[ndarray | None],
    train_dataset: Subset,
    test_dataset: Subset,
    p: float = 0.5,
    epochs: int = 1,
    batch_size: int = 1,
    seed: int | Generator | None = None
):
    """
    A
    """
    if isinstance(seed, Generator):
        generator = seed
    else:
        seeder = SeedSequence()
        if isinstance(seed, int):
            seeder = SeedSequence(seed)
        generator = numpyrand.default_rng(PCG64(seeder))
    fittner = WeightFitnessCalculator()
    fittner.set_params(
        train_dataset,
        test_dataset,
        epochs,
        batch_size
    )
    best_fitness, best_weights = fittner.evaluate(weights)
    weights, prev_fitness = best_weights, best_fitness
    temperature = t_i
    for _ in range(iterations):
        neighbors = get_neighbors(best_weights, p, generator)
        neighbor = generator.choice(neighbors)
        fitness, neighbor = fittner.evaluate(neighbor, seed=generator)
        if fitness > best_fitness:
            best_fitness, best_weights = fitness, neighbor
        if bool(generator.binomial(
            1, min(1, numpy.exp(-(prev_fitness - fitness) / temperature))
        )):
            weights, prev_fitness = neighbor, fitness
        temperature *= t_decay
    return best_weights
