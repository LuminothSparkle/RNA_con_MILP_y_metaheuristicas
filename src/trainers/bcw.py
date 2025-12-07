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
from utility.io.nnjson import gen_from_tuple, crossvalidator_json, dataset_json, trainer_json, arch_json
from utility.io.crossvalidation import save_crossvalidation, load_crossvalidation
from utility.nn.lineal import LinealNN
from utility.signal import InterruptHandler


def load_crossvalidator(load_path: Path, generate: bool = False):
    if generate:
        dataset_kwargs = dataset_json(load_path / 'dataset.json')
        crossvalidator_kwargs = crossvalidator_json(load_path / 'crossvalidation.json')
        crossvalidator_kwargs['crossvalidator'] = gen_from_tuple(crossvalidator_kwargs['crossvalidator'])
        arch_kwargs = arch_json(load_path / 'architecture.json')
        trainer_kwargs = trainer_json(load_path / 'trainer.json')
        dataset = bcw_dataset(**dataset_kwargs)
        arch_kwargs['capacity'] = [
            sum(dataset.get_tensor_sizes('features')),
            *arch_kwargs['capacity'],
            sum(dataset.get_tensor_sizes('targets'))
        ]
        arch_kwargs['inference_layer'] = dataset.inference_function
        arch_kwargs['loss_layer'] = dataset.loss_fn
        crossvalidator = CrossvalidationNN.from_dataset(
            dataset=dataset,
            base_model=LinealNN.from_capacity(**arch_kwargs),
            optimizer=trainer_kwargs['optimizer'][0],
            optimizer_kwargs=trainer_kwargs['optimizer'][2],
            scheculer=trainer_kwargs['scheduler'][0],
            scheduler_kwargs=trainer_kwargs['scheduler'][2],
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
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(crossvalidator.crossvalidate)
        with InterruptHandler() as handler:
            while not future.done():
                sleep(1)
                crossvalidator.interrupted = handler.interrupted
            if handler.interrupted:
                print("Interrupted")
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
