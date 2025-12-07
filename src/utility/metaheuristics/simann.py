"""
A
"""
import numpy
import numpy.random as numpyrand
from numpy import ndarray
from numpy.random import PCG64, SeedSequence, Generator
from utility.nn.dataset import CsvDataset
from utility.nn.trainer import TrainerNN
from utility.metaheuristics.fitness import WeightFitnessCalculator
from utility.metaheuristics.nnnbhd import get_neighbors


def simulated_annealing(
    trainer: TrainerNN,
    dataset: CsvDataset,
    t_i: float,
    iterations: int, t_decay: float,
    weights: list[ndarray | None],
    p: float = 0.5,
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
    fittner = WeightFitnessCalculator(
        arch=trainer.model,
        trainer=trainer,
        dataset=dataset
    )
    best_fitness, best_weights = fittner.evaluate(weights)
    weights, prev_fitness = best_weights, best_fitness
    temperature = t_i
    for _ in range(iterations):
        neighbors = get_neighbors(best_weights, p, generator)
        neighbor = generator.choice(neighbors)
        fitness, neighbor = fittner.evaluate(neighbor)
        if fitness > best_fitness:
            best_fitness, best_weights = fitness, neighbor
        if bool(generator.binomial(
            1, min(1, numpy.exp(-(prev_fitness - fitness) / temperature))
        )):
            weights, prev_fitness = neighbor, fitness
        temperature *= t_decay
    return best_weights
