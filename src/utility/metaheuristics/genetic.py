"""
A
"""
from concurrent.futures import ThreadPoolExecutor
from typing import Callable
import numpy
from numpy.random import (
    Generator, PCG64
)
import torch
from torch import Tensor


def uniform_crossover(
    chromosome_a: Tensor, chromosome_b: Tensor,
    p: float = 0.5, generator: torch.Generator | int | None = None
):
    """
    A
    """
    if generator is None:
        generator = torch.Generator(torch.get_default_device())
    elif isinstance(generator, int):
        seed = generator
        generator = torch.Generator(torch.get_default_device())
        generator.manual_seed(seed)
    mask = torch.rand_like(chromosome_a).bernoulli(
        p=p,
        generator=generator
    ).bool()
    ind_a = chromosome_a.bool().bitwise_and(mask).bitwise_or(
        chromosome_b.bool().bitwise_and(mask.bitwise_not())
    ).bool()
    ind_b = chromosome_a.bool().bitwise_and(mask.bitwise_not()).bitwise_or(
        chromosome_b.bool().bitwise_and(mask)
    ).bool()
    return ind_a, ind_b


def kpoint_crossover(
    chromosome_a: Tensor, chromosome_b: Tensor,
    k: int = 1, generator: torch.Generator | int | None = None
):
    """
    A
    """
    if generator is None:
        generator = torch.Generator(torch.get_default_device())
    elif isinstance(generator, int):
        seed = generator
        generator = torch.Generator(torch.get_default_device())
        generator.manual_seed(seed)
    mask = (
        torch.arange(chromosome_a.numel()).t()
        < torch.randperm(
            chromosome_a.numel(),
            generator=generator
        )[:k].unsqueeze(dim=0)
    ).sum(dim=0).remainder(2).bool()
    ind_a = chromosome_a.bool().bitwise_and(mask).bitwise_or(
        chromosome_b.bool().bitwise_and(mask.bitwise_not())
    ).bool()
    ind_b = chromosome_a.bool().bitwise_and(mask.bitwise_not()).bitwise_or(
        chromosome_b.bool().bitwise_and(mask)
    ).bool()
    return ind_a, ind_b


def mutation(
    chromosome: Tensor, p: float = 0.01,
    generator: torch.Generator | int | None = None
):
    """
    A
    """
    if generator is None:
        generator = torch.Generator(torch.get_default_device())
    elif isinstance(generator, int):
        seed = generator
        generator = torch.Generator(torch.get_default_device())
        generator.manual_seed(seed)
    return chromosome.bool().bitwise_xor(
        torch.rand_like(chromosome).bernoulli(
            p=p,
            generator=generator
        ).bool()
    )


def random_poblation(
    base_chromosome: Tensor,
    size: int,
    generator: torch.Generator | int | None = None
):
    """
    A
    """
    if generator is None:
        generator = torch.Generator(torch.get_default_device())
    elif isinstance(generator, int):
        seed = generator
        generator = torch.Generator(torch.get_default_device())
        generator.manual_seed(seed)
    poblation = []
    for _ in range(size):
        new_chromosome = base_chromosome.clone()
        new_chromosome[torch.randperm(
            base_chromosome.numel(),
            generator=generator
        )[:(base_chromosome.numel() // size)]] = 0
        poblation += [new_chromosome]
    return poblation


def genetic_loop(
    base_chromosome: Tensor,
    poblation_size: int,
    generations: int,
    crossover_function: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]],
    mutation_function: Callable[[Tensor], Tensor],
    aptitude_function: Callable[[Tensor], float],
    parent_selection_function: Callable[[list[float]], list[tuple[int, int]]],
    selection_function: Callable[
        [list[float], list[float]],
        tuple[list[int], list[int]]
    ],
    generator: torch.Generator | int | None = None
):
    """
    A
    """
    poblation = random_poblation(base_chromosome, poblation_size, generator)
    fitness = [
        aptitude_function(ind)
        for ind in poblation
    ]
    best_i = numpy.argmin(fitness)
    best = {
        'aptitude': fitness[best_i],
        'individual': poblation[best_i],
        'generations': [fitness[best_i]]
    }
    for generation in range(generations):
        print(f'Procesando generacion {generation}')
        parent_selected = parent_selection_function(fitness)
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures_list = []

            def gen_individuals(a: int, b: int):
                ind_a, ind_b = crossover_function(poblation[a], poblation[b])
                ind_a, ind_b = (
                    mutation_function(ind)
                    for ind in (ind_a, ind_b)
                )
                return ind_a, ind_b
            for a, b in parent_selected:
                futures_list += [executor.submit(
                    gen_individuals, a, b
                )]
            new_poblation = [
                ind
                for tup in (
                    future_data.result()
                    for future_data in futures_list
                )
                for ind in tup
            ]
            futures_list = []
            for ind in new_poblation:
                futures_list += [executor.submit(
                    aptitude_function,
                    ind
                )]
            new_fitness = [
                future_data.result()
                for future_data in futures_list
            ]
        best_i = numpy.argmin(new_fitness)
        if best['aptitude'] > new_fitness[best_i]:
            best['aptitude'] = new_fitness[best_i]
            best['individual'] = new_poblation[best_i]
        best['generations'] += [best['aptitude']]
        old_list, new_list = selection_function(fitness, new_fitness)
        poblation = [
            *[poblation[old] for old in old_list],
            *[new_poblation[new] for new in new_list]
        ]
        fitness = [
            *[fitness[old] for old in old_list],
            *[new_fitness[new] for new in new_list]
        ]
    return best


def elitist_selection(
    fitness: list[float], survivors: int
):
    """
    A
    """
    return numpy.argsort(
        fitness
    )[-1:][0:survivors].tolist()


def parent_elitist_selection(fitness: list[float], pairs: int):
    """
    A
    """
    parent_list = elitist_selection(fitness, pairs * 2)
    return [*zip(parent_list[0::2], parent_list[1::2])]


def poblation_elitist_selection(
    old_fitness: list[float],
    new_fitness: list[float],
    survivors: int
):
    """
    A
    """
    all_fitness = [
        *old_fitness,
        *new_fitness
    ]
    survivor_list = elitist_selection(all_fitness, survivors)
    survivor_old = [
        survivor for survivor in survivor_list if survivor < len(old_fitness)
    ]
    survivor_new = [
        survivor for survivor in survivor_list if survivor >= len(old_fitness)
    ]
    return survivor_old, survivor_new


def rank_selection(
    fitness: list[float], survivors: int,
    pressure: float | None = None,
    exp_sampling: bool = False,
    generator: Generator | int | None = None
):
    """
    A
    """
    if generator is None:
        generator = Generator(PCG64())
    elif isinstance(generator, int):
        generator = Generator(PCG64(generator))
    ranking = numpy.argsort(fitness)
    total_individuals = len(fitness)
    if exp_sampling:
        if pressure is None:
            pressure = 0.5
        weights = numpy.pow(pressure, numpy.arange(1, total_individuals + 1))
        cum_prob = (weights / weights.sum()).cumsum()
    else:
        if pressure is None:
            pressure = 1.0
        cum_prob = (1 / total_individuals * (
            pressure - (2 * pressure - 2)
            * numpy.arange(0, total_individuals)
        )).cumsum()
    pointers = generator.uniform(0, 1, survivors)
    return ranking[
        numpy.searchsorted(a=cum_prob, v=pointers, side='left')
    ].tolist()


def parent_rank_selection(
    fitness: list[float], pairs: int,
    pressure: float | None = None,
    exp_sampling: bool = False,
    generator: Generator | int | None = None
):
    """
    A
    """
    parent_list = rank_selection(
        fitness, pairs * 2, pressure, exp_sampling, generator
    )
    return [*zip(parent_list[0::2], parent_list[1::2])]


def poblation_rank_selection(
    old_fitness: list[float],
    new_fitness: list[float],
    survivors: int,
    pressure: float | None = None,
    exp_sampling: bool = False,
    generator: Generator | int | None = None
):
    """
    A
    """
    all_fitness = [
        *old_fitness,
        *new_fitness
    ]
    survivor_list = rank_selection(
        all_fitness, survivors, pressure, exp_sampling, generator
    )
    survivor_old = [
        survivor for survivor in survivor_list if survivor < len(old_fitness)
    ]
    survivor_new = [
        survivor for survivor in survivor_list if survivor >= len(old_fitness)
    ]
    return survivor_old, survivor_new


def roulette_wheel_selection(
    fitness: list[float], survivors: int,
    su_samping: bool = False,
    generator: Generator | int | None = None
):
    """
    A
    """
    if generator is None:
        generator = Generator(PCG64())
    elif isinstance(generator, int):
        generator = Generator(PCG64(generator))
    cum_fitness = numpy.cumsum(fitness)
    total_fitness = cum_fitness[-1]
    if su_samping:
        interval = total_fitness / survivors
        pointers = numpy.arange(
            generator.uniform(0, interval),
            total_fitness,
            interval
        )
    else:
        pointers = generator.uniform(0, total_fitness, survivors)
    return numpy.searchsorted(a=cum_fitness, v=pointers, side='left').tolist()


def parent_roulette_wheel_selection(
    fitness: list[float], pairs: int,
    su_samping: bool = False,
    generator: Generator | int | None = None
):
    """
    A
    """
    parent_list = roulette_wheel_selection(
        fitness, pairs * 2, su_samping, generator
    )
    return [*zip(parent_list[0::2], parent_list[1::2])]


def poblation_roulette_wheel_selection(
    old_fitness: list[float],
    new_fitness: list[float],
    survivors: int,
    su_sampling: bool = False,
    generator: Generator | int | None = None
):
    """
    A
    """
    all_fitness = [
        *old_fitness,
        *new_fitness
    ]
    survivor_list = roulette_wheel_selection(
        all_fitness, survivors, su_sampling, generator
    )
    survivor_old = [
        survivor for survivor in survivor_list if survivor < len(old_fitness)
    ]
    survivor_new = [
        survivor for survivor in survivor_list if survivor >= len(old_fitness)
    ]
    return survivor_old, survivor_new
