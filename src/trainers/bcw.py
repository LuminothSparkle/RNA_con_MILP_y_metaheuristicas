"""
Codigo que entrena la red neuronal de Breast Cancer Winsconsin Diagnostic
"""
import argparse
from argparse import ArgumentParser
from pathlib import Path
from time import sleep
from concurrent.futures import ThreadPoolExecutor
from trainers.datasets import bcw_dataset
from utility.nn.crossvalidation import CrossvalidationNN
from utility.io.nnjson import gen_from_tuple, read_nn_json
from utility.io.crossvalidation import save_crossvalidation, load_crossvalidation
from utility.nn.lineal import LinealNN
from utility.signal import InterruptHandler


def load_crossvalidator(load_path: Path, generate: bool = False):
    if generate:
        dataset_kwargs = read_nn_json(load_path / 'dataset.json')
        crossvalidator_kwargs = read_nn_json(load_path / 'crossvalidation.json')
        crossvalidator_kwargs['crossvalidator'] = gen_from_tuple(
            crossvalidator_kwargs['crossvalidator']
        )
        arch_kwargs = read_nn_json(load_path / 'architecture.json')
        trainer_kwargs = read_nn_json(load_path / 'trainer.json')
        dataset = bcw_dataset(**dataset_kwargs)
        trainer_kwargs['batch_size'] = (
            len(dataset.train_dataframe) // trainer_kwargs['train_batches']
        )
        trainer_kwargs.pop('train_batches')
        arch_kwargs['capacity'] = [
            sum(dataset.get_tensor_sizes('features')),
            *arch_kwargs['capacity'],
            sum(dataset.get_tensor_sizes('targets'))
        ]
        crossvalidator = CrossvalidationNN.from_dataset(
            dataset=dataset,
            base_model=LinealNN(**arch_kwargs),
            optimizer=trainer_kwargs['optimizer'][0],
            optimizer_kwargs=trainer_kwargs['optimizer'][2],
            scheculer=trainer_kwargs['scheduler'][0],
            scheduler_kwargs=trainer_kwargs['scheduler'][2],
            batch_size=trainer_kwargs['batch_size'],
            epochs=trainer_kwargs['epochs'],
            **crossvalidator_kwargs
        )
    else:
        crossvalidator = load_crossvalidation(load_path)
    return crossvalidator

def main(args: argparse.Namespace):
    """
    Funcion main para generar la base de datos en archivo csv del archivo
    json del kardex anonimizado
    """
    crossvalidator = load_crossvalidator(args.load_path, not args.load)
    switch = False
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(crossvalidator.crossvalidate)
        with InterruptHandler() as handler:
            while not future.done():
                sleep(5)
                crossvalidator.interrupted = handler.interrupted
                if not switch and handler.interrupted:
                    print("Interrupted")
                    switch = True
        if future.result():
            print("Terminó")
        else:
            print("No terminó")
    save_crossvalidation(crossvalidator, args.save_path)


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
