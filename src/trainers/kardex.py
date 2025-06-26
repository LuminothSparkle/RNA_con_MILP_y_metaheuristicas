"""
Codigo que entrena las redes neuronales del problema del problema de
predecir si un estudiante egresará o abandonará de acuerdo a su historial
académico
"""
import argparse
from argparse import ArgumentParser
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import torch
from src.trainers.datasets import KardexDataset
from src.utility.nn.cvdataset import crossvalidate
from src.utility.io.nnjson import read_arch_json, read_cv_json
from src.crossvalidation.cvdir import save_crossvalidation, generate_cv_data


def test_arch(dataset: KardexDataset, arch_data: dict, cv_data: dict):
    """
    A
    """
    torch.manual_seed(arch_data['torch seed'])
    loss_fn = dataset.loss_fn
    base_cv, cv_kwargs = cv_data['crossvalidator']
    cv = base_cv(**cv_kwargs)
    return crossvalidate(
        arch=arch_data['capacity'],
        dataset=dataset,
        optimizer=arch_data['optimizer'],
        loss_fn=loss_fn,
        epochs=arch_data['epochs'],
        iterations=cv_data['iterations'],
        crossvalidator=cv,
        train_batches=cv_data['train batches'],
        extra_params=arch_data
    )


def process_career(
    career_path: Path, save_path: Path,
    arch_data: dict, cv_data: dict, gen_data: bool = False,
    exists_ok: bool = True
):
    """
    A
    """
    print(f'Processing {career_path.stem}')
    dataset = KardexDataset(career_path, cv_data)
    arch_data['capacity'] = [
        dataset.features_size,
        *arch_data['capacity'],
        dataset.targets_size
    ]
    data = test_arch(dataset, arch_data, cv_data)
    name = career_path.stem
    results_path = save_path / name / 'results'
    results_path.mkdir(parents=True, exist_ok=True)
    save_crossvalidation(results_path, data, name, exists_ok)
    if gen_data:
        generate_cv_data(
            results_path,
            data,
            'kardex',
            exists_ok
        )


def main(args: argparse.Namespace):
    """
    Funcion main para generar la base de datos en archivo csv del archivo
    json del kardex anonimizado
    """
    load_path = Path(args.load_path)
    save_path = Path(args.save_path)
    arch_path = Path(args.arch_path)
    cv_path = Path(args.cv_path)
    if not load_path.exists():
        print(f'{load_path} doesn\'t exists')
        return None
    elif not load_path.is_dir():
        print(f'Cannot access {load_path} or isn\'t a directory')
        return None
    cv_data = read_cv_json(cv_path)
    arch_data = read_arch_json(arch_path)
    file_list = list(load_path.glob('*.csv'))
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures_list = []
        for career_path in file_list:
            futures_list += [executor.submit(
                process_career,
                career_path,
                save_path,
                arch_data,
                cv_data,
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
        default=Path.cwd() / 'Data' / 'kardex'
    )
    argparser.add_argument(
        '--load_path', '-lp',
        default=Path.cwd() / 'Data' / 'kardex'
    )
    argparser.add_argument(
        '--arch_path', '-ap',
        default=Path.cwd() /
        'Data' / 'kardex' /
        'crossvalidation' /
        'arch.json'
    )
    argparser.add_argument(
        '--cv_path', '-cp',
        default=Path.cwd() /
        'Data' / 'kardex' /
        'crossvalidation' /
        'cv.json'
    )
    argparser.add_argument('--no_overwrite', '-no', action='store_true')
    argparser.add_argument('--gen_data', '-gd', action='store_true')
    torch.set_default_device('cpu')
    torch.set_default_dtype(torch.double)
    if torch.cuda.is_available():
        torch.set_default_device('cuda')
        torch.set_default_dtype(torch.double)
    sys.exit(main(argparser.parse_args(sys.argv[1:])))
