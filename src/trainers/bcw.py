"""
Codigo que entrena la red neuronal de Breast Cancer Winsconsin Diagnostic
"""
import argparse
from argparse import ArgumentParser
from pathlib import Path
import torch
from src.trainers.datasets import BCWDataset, test_arch
from src.utility.io.nnjson import gen_from_tuple, read_arch_json, read_cv_json
from src.crossvalidation.cvdir import generate_cv_data
from src.utility.io.crossvalidation import save_crossvalidation


def main(args: argparse.Namespace):
    """
    Funcion main para generar la base de datos en archivo csv del archivo
    json del kardex anonimizado
    """
    torch.set_default_device('cpu')
    torch.set_default_dtype(torch.double)
    if torch.cuda.is_available():
        torch.set_default_device('cuda')
        torch.set_default_dtype(torch.double)
    cv_data = read_cv_json(args.cv_path)
    arch_data = read_arch_json(args.arch_path)
    cv_data['crossvalidator'] = gen_from_tuple(cv_data['crossvalidator'])
    dataset = BCWDataset(**cv_data)
    arch_data['capacity'] = [
        dataset.features_size,
        *arch_data['capacity'],
        dataset.targets_size
    ]
    arch_data['inference_layer'] = dataset.inference_function
    arch_data['loss_layer'] = dataset.loss_fn
    if 'threads' in cv_data:
        arch_data['threads'] = cv_data['threads']
    results_dict = test_arch(
        dataset=dataset,
        iterations=cv_data['iterations'],
        train_batches=cv_data['train_batches'],
        **arch_data
    )
    results_path = args.save_path / 'results'
    results_path.mkdir(parents=True, exist_ok=True)
    save_crossvalidation(
        file_path=results_path / 'crossvalidation.pt',
        exists_ok=not args.no_overwrite,
        **results_dict
    )
    if args.gen_data:
        generate_cv_data(
            dir_path=results_path,
            name='bcw',
            exists_ok=not args.no_overwrite,
            **results_dict
        )


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
    argparser.add_argument(
        '--arch_path', '-ap',
        type=Path,
        default=Path.cwd() / 'Data' /
        'Breast Cancer Winsconsin (Diagnostic)' /
        'crossvalidation' / 'arch.json'
    )
    argparser.add_argument(
        '--cv_path', '-cp',
        type=Path,
        default=Path.cwd() / 'Data' /
        'Breast Cancer Winsconsin (Diagnostic)' /
        'crossvalidation' / 'cv.json'
    )
    argparser.add_argument('--no_overwrite', '-no', action='store_true')
    argparser.add_argument('--gen_data', '-gd', action='store_true')
    sys.exit(main(argparser.parse_args(sys.argv[1:])))
