"""
Codigo que entrena la red neuronal de Breast Cancer Winsconsin Diagnostic
"""
import argparse
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path
from time import sleep
from concurrent.futures import ThreadPoolExecutor
from utility.signal import InterruptHandler
import numpy
import utility.nn.torchdefault as torchdefault
from utility.io.crossvalidation import load_trainer, load_dataset
from utility.metaheuristics.genetic import (
    genetic_loop, uniform_crossover, mutation, parent_roulette_wheel_selection,
    parent_rank_selection, parent_elitist_selection, poblation_roulette_wheel_selection,
    poblation_elitist_selection, poblation_rank_selection, get_chromosome
)
from utility.metaheuristics.fitness import (
    MaskFitnessCalculator
)
from utility.io.metaheuristics import save_archs
from utility.nn.stats import comparative_dataframe, compare_archs


def main(args: argparse.Namespace):
    """
    Funcion main para generar la base de datos en archivo csv del archivo
    json del kardex anonimizado
    """
    torchdefault.set_defaults()
    trainer = load_trainer(args.load_path)
    trainer.epochs = 100
    trainer.overfit_tolerance = 10
    base_chromosome = get_chromosome(trainer.model.get_weights(), 0.1)
    fc = MaskFitnessCalculator(archs=trainer.model, trainer=trainer, dataset=trainer.dataset)
    ss = numpy.random.SeedSequence(0)
    generator = numpy.random.default_rng(ss)
    torch_generator = torchdefault.generator()
    crossover_fn = lambda a,b : uniform_crossover(a, b, 0.5, torch_generator)
    mutation_fn = lambda a : mutation(a, 0.01, torch_generator)
    aptitude_fn = lambda a : fc.evaluate(a)
    par_sel_fn = lambda fit : parent_rank_selection(fit, 3, 0.1, True, generator)
    sel_fn = lambda fit1,fit2 : poblation_rank_selection(fit1, fit2, 6, 0.1, True, generator)
    best = genetic_loop(
        base_chromosome, 6, 6, crossover_fn, mutation_fn,
        aptitude_fn, par_sel_fn, sel_fn, torch_generator
    )
    new_arch = fc.get_best_arch(best['individual'])['arch']
    new_trainer = deepcopy(trainer)
    new_trainer.set_model(new_arch)
    save_archs(trainer, new_trainer, args.save_path)
    comparative_dataframe(
        [trainer, new_trainer],
        ['original', 'optimized']
    ).to_csv(args.save_path / 'genetic.csv')


if __name__ == '__main__':
    import sys
    argparser = ArgumentParser()
    argparser.add_argument(
        '--save_path', '-sp',
        type=Path,
        default=Path.cwd() / 'Data' /
        'Breast Cancer Winsconsin (Diagnostic)'
    )
    argparser.add_argument(
        '--load_path', '-lp',
        type=Path,
        default=Path.cwd() / 'Data' /
        'Breast Cancer Winsconsin (Diagnostic)'
    )
    argparser.add_argument('--load', '-l', action='store_true')
    sys.exit(main(argparser.parse_args(sys.argv[1:])))
