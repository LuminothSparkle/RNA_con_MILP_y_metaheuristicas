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
from utility.io.crossvalidation import load_trainer
from utility.metaheuristics.hill_climb import hill_climb
from utility.io.metaheuristics import save_archs


def main(args: argparse.Namespace):
    """
    Funcion main para generar la base de datos en archivo csv del archivo
    json del kardex anonimizado
    """
    torchdefault.set_defaults()
    trainer = load_trainer(args.load_path)
    ss = numpy.random.SeedSequence(0)
    generator = numpy.random.default_rng(ss)
    new_weights = hill_climb(
        weights=trainer.model.get_weights(),
        dataset=trainer.dataset,
        trainer=trainer,
        p=0.5,
        seed=generator
    )
    new_trainer = deepcopy(trainer)
    new_trainer.model.set_weights(new_weights)
    save_archs(trainer, new_trainer, args.save_path)

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
