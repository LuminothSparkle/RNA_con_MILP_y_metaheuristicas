"""
A
"""
from numpy import ndarray
import numpy.random as numpyrand
from numpy.random import Generator, SeedSequence, PCG64
from .nnred import get_capacity
from utility.nn.trainer import TrainerNN
from utility.nn.dataset import CsvDataset
from utility.metaheuristics.fitness import WeightFitnessCalculator
from utility.metaheuristics.nnnbhd import get_neighbors


def hill_climb(
    weights: list[ndarray | None],
    dataset: CsvDataset,
    trainer: TrainerNN,
    p: float = 0.5,
    seed: int | Generator | None = None,
    iterations: int | None = None,
    tolerance: float | None = None
):
    """
    A
    """
    if tolerance is None:
        tolerance = 0.05
    if iterations is None:
        iterations = 10
    if isinstance(seed, Generator):
        generator = seed
    else:
        seeder = SeedSequence()
        if isinstance(seed, int):
            seeder = SeedSequence(seed)
        generator = numpyrand.default_rng(PCG64(seeder))
    fittner = WeightFitnessCalculator(
        arch=trainer.model, trainer=trainer,
        dataset=dataset
    )
    best_fitness, best_weights = fittner.evaluate(weights)
    init_fitness = best_fitness
    for iteration in range(iterations):
        print(f"iteracion {iteration}")
        print(f"Solución actual {get_capacity(best_weights)[1:-1]}")
        print(f"Fitness {best_fitness}")
        neighbors = get_neighbors(best_weights, p, generator)
        fitnesses = []
        new_neighbors = []
        for i, neighbor in enumerate(neighbors):
            print(f"Evaluando vecino {get_capacity(neighbor)[1:-1]}")
            fitness, new_neighbor = fittner.evaluate(neighbor)
            fitnesses += [(fitness, i)]
            print(f"Fitness {fitness}")
            new_neighbors += [new_neighbor]
        sorted_fitness = sorted(fitnesses)
        if init_fitness > sorted_fitness[-1][0] * (1 +  tolerance):
            break
        best_fitness = sorted_fitness[-1][0]
        best_weights = new_neighbors[sorted_fitness[-1][1]]
    return best_weights
