"""
A
"""
from itertools import accumulate
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
from argparse import ArgumentParser
from pandas import read_csv
from crossvalidation.files import safe_suffix


def generate_batches(
    load_file: Path,
    save_path: Path,
    batch_size: int = 1,
    exists_ok: bool = True
):
    """
    A
    """
    dataframe = read_csv(load_file, header=0, index_col=0)
    batches = len(dataframe) // batch_size
    sizes = [
        batch_size + (batch < len(dataframe) % batches)
        for batch in range(batches)
    ]
    indices = list(accumulate([0, *sizes]))
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures_list = []
        for batch in range(batches):
            dir_path = (
                save_path / f'batch_{batch}'
            )
            dir_path.mkdir(parents=True, exist_ok=True)
            file_path = dir_path / f'{load_file.stem}.csv'
            assert exists_ok or not file_path.exists(), (
                f"El archivo {file_path} ya existe"
            )
            futures_list += [executor.submit(
                dataframe.iloc[
                    indices[batch]:indices[batch + 1],
                    :
                ].to_csv,
                file_path,
                header=True,
                index=True,
                index_label='ID'
            )]
        for future_data in futures_list:
            future_data.result()


def main(args: argparse.Namespace):
    """
    Funcion main para generar la base de datos en archivo csv del archivo
    json del kardex anonimizado
    """
    extensions = ['ftr', 'reg_tgt', 'cls_tgt']
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures_list = []
        csv_dir_name = safe_suffix('model', args.case_index)
        for extension in extensions:
            dir_path = (
                args.load_path / csv_dir_name / 'train_csv'
            )
            futures_list += [executor.submit(
                generate_batches,
                dir_path / f'{args.load_name}_{extension}.csv',
                args.save_path / csv_dir_name / 'train_csv',
                args.batch_size,
                not args.no_overwrite
            )]
        for future_data in futures_list:
            future_data.result()


if __name__ == '__main__':
    import sys
    argparser = ArgumentParser()
    argparser.add_argument(
        '--save_path', '-sp',
        type=Path, default=Path.cwd()
    )
    argparser.add_argument(
        '--load_path', '-lp',
        type=Path, default=Path.cwd()
    )
    argparser.add_argument(
        '--load_name', '-ln',
        type=str, default=''
    )
    argparser.add_argument(
        '--case_index', '-ci',
        type=str, default=''
    )
    argparser.add_argument(
        '--no_overwrite', '-eo',
        action='store_true'
    )
    argparser.add_argument(
        '--batch_size', '-bs',
        type=int, default=10
    )
    sys.exit(main(argparser.parse_args(sys.argv[1:])))
