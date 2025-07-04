"""
Codigo que entrena las redes neuronales del problema del problema de
predecir si un estudiante egresará o abandonará de acuerdo a su historial
académico
"""
import argparse
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import torch
from src.trainers.datasets import KardexDataset, test_arch
from src.utility.io.nnjson import read_arch_json, read_cv_json, gen_from_tuple
from src.crossvalidation.cvdir import generate_cv_data
from src.utility.io.crossvalidation import save_crossvalidation


def process_career(
    career_path: Path, save_path: Path,
    arch_data: dict, cv_data: dict, gen_data: bool = False,
    exists_ok: bool = True
):
    """
    A
    """
    cv_data = deepcopy(cv_data)
    torch.set_default_device('cpu')
    torch.set_default_dtype(torch.double)
    if torch.cuda.is_available():
        torch.set_default_device('cuda')
        torch.set_default_dtype(torch.double)
    print(f'Processing {career_path.stem}')
    cv_data['crossvalidator'] = gen_from_tuple(cv_data['crossvalidator'])
    dataset = KardexDataset(career_path, **cv_data)
    arch_data['inference_layer'] = dataset.inference_function
    arch_data['loss_layer'] = dataset.loss_fn
    arch_data['capacity'] = [
        dataset.features_size,
        *arch_data['capacity'],
        dataset.targets_size
    ]
    if 'threads' in cv_data:
        arch_data['threads'] = cv_data['threads']
    data = test_arch(
        dataset=dataset,
        iterations=cv_data['iterations'],
        train_batches=cv_data['train_batches'],
        **arch_data
    )
    name = career_path.stem
    results_path = save_path / name / 'results'
    results_path.mkdir(parents=True, exist_ok=True)
    save_crossvalidation(
        file_path=results_path / 'crossvalidation.pt',
        exists_ok=exists_ok,
        **data
    )
    if gen_data:
        generate_cv_data(
            dir_path=results_path,
            name='kardex',
            exists_ok=exists_ok,
            **data
        )


def main(args: argparse.Namespace):
    """
    Funcion main para generar la base de datos en archivo csv del archivo
    json del kardex anonimizado
    """
    cv_data = read_cv_json(args.cv_path)
    arch_data = read_arch_json(args.arch_path)
    file_list = list(args.load_path.glob('*.csv'))
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures_list = []
        for career_path in file_list:
            futures_list += [executor.submit(
                process_career,
                career_path,
                args.save_path,
                deepcopy(arch_data),
                deepcopy(cv_data),
                args.gen_data,
                not args.no_overwrite
            )]
        for future_data in futures_list:
            future_data.result()


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
        '--arch_path', '-ap',
        type=Path,
        default=Path.cwd() /
        'Data' / 'kardex' /
        'crossvalidation' /
        'arch.json'
    )
    argparser.add_argument(
        '--cv_path', '-cp',
        type=Path,
        default=Path.cwd() /
        'Data' / 'kardex' /
        'crossvalidation' /
        'cv.json'
    )
    argparser.add_argument('--no_overwrite', '-no', action='store_true')
    argparser.add_argument('--gen_data', '-gd', action='store_true')
    sys.exit(main(argparser.parse_args(sys.argv[1:])))
