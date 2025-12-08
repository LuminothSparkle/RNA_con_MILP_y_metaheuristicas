"""
Codigo que entrena las redes neuronales del problema del problema de
predecir si un estudiante egresará o abandonará de acuerdo a su historial
académico
"""
import argparse
from argparse import ArgumentParser
from pathlib import Path
from time import sleep
from concurrent.futures import ThreadPoolExecutor
from trainers.datasets import kardex_dataset
from utility.nn.crossvalidation import CrossvalidationNN
from utility.io.nnjson import gen_from_tuple, read_nn_json
from utility.io.crossvalidation import save_crossvalidation, load_crossvalidation
from utility.nn.lineal import LinealNN
from utility.signal import InterruptHandler


def load_crossvalidator(load_path: Path, data_path: Path, career_name: str, generate: bool = False):
    if generate:
        dataset_kwargs = read_nn_json(load_path / 'dataset.json')
        crossvalidator_kwargs = read_nn_json(load_path / 'crossvalidation.json')
        crossvalidator_kwargs['crossvalidator'] = gen_from_tuple(crossvalidator_kwargs['crossvalidator'])
        arch_kwargs = read_nn_json(load_path / 'architecture.json')
        trainer_kwargs = read_nn_json(load_path / 'trainer.json')
        dataset = kardex_dataset(file_path=data_path / career_name, **dataset_kwargs)
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
    career_paths = list(args.data_path.glob('*.csv'))
    crossvalidators = []
    for career_path in career_paths:
        career_name = career_path.name
        crossvalidators += [load_crossvalidator(
            args.load_path, args.data_path, career_name, not args.load
        )]
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures_list = []
        for crossvalidator in crossvalidators:
            futures_list += [
                executor.submit(crossvalidator.crossvalidate)
            ]
        switch = False
        with InterruptHandler() as handler:
            while not all(future.done() for future in futures_list):
                sleep(5)
                for crossvalidator in crossvalidators:
                    crossvalidator.interrupted = handler.interrupted
                    if not switch and handler.interrupted:
                        print("Interrupted")
                        switch = True
        if all(future.result() for future in futures_list):
            print("Terminó")
        else:
            print("No terminó")
    for crossvalidator, career_path in zip(crossvalidators, career_paths):
        career_name = career_path.stem
        save_crossvalidation(crossvalidator, args.save_path / career_name)


if __name__ == '__main__':
    import sys
    argparser = ArgumentParser()
    argparser.add_argument(
        '--save_path', '-sp',
        type=Path,
        default=Path.cwd() / 'Data' / 'kardex'
    )
    argparser.add_argument(
        '--load_path', '-lp',
        type=Path,
        default=Path.cwd() / 'Data' / 'kardex'
    )
    argparser.add_argument(
        '--data_path', '-dp',
        type=Path,
        default=Path.cwd() / 'Data' / 'kardex'
    )
    argparser.add_argument('--load', '-l', action='store_true')
    sys.exit(main(argparser.parse_args(sys.argv[1:])))
