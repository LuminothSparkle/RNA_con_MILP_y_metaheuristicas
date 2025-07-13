"""
A
"""
from concurrent.futures import ThreadPoolExecutor
from numpy import ndarray
import numpy
import numpy.random as numpyrand
from numpy.random import Generator, SeedSequence, PCG64
from torch.utils.data import Subset
from src.utility.metaheuristics.fitness import WeightFitnessCalculator
from src.utility.metaheuristics.nnnbhd import get_neighbors


def hill_climb(
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
    while True:
        neighbors = get_neighbors(weights, p, generator)
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures_list = []
            for neighbor in neighbors:
                futures_list += [executor.submit(
                    fittner.evaluate,
                    weights=neighbor,
                    seed=generator
                )]
            neighbors = []
            fitnesses = []
            for future_data in futures_list:
                fitness, neighbor = future_data.result()
                fitnesses += [fitness]
                neighbors += [neighbor]
        max_i = numpy.argmax(fitnesses)
        if best_fitness > fitnesses[max_i]:
            break
        best_fitness = fitnesses[max_i]
        best_weights = neighbors[max_i]
    return best_weights
