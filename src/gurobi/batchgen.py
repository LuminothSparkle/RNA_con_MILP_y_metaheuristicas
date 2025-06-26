"""
A
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
from argparse import ArgumentParser
from pandas import read_csv


def main(args: argparse.Namespace):
    """
    Funcion main para generar la base de datos en archivo csv del archivo
    json del kardex anonimizado
    """
    dataframe = read_csv(args.load_file, header=0, index_col=0)
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures_list = []
        for batch, start in enumerate(range(
            0, len(dataframe) + args.batch_size, args.batch_size
        )):
            file_path = args.save_path / f'{args.load_file.stem()}_{batch}'
            assert not args.no_overwrite() or not file_path.exists(), (
                f"El archivo {file_path} ya existe"
            )
            futures_list += [executor.submit(
                dataframe[
                    start:(min(start + args.batch_size, len(dataframe))),
                    :
                ].to_csv,
                file_path,
                header=True,
                index=True,
                index_label='ID'
            )]
        for future_data in futures_list:
            future_data.result()


if __name__ == '__main__':
    import sys
    argparser = ArgumentParser()
    argparser.add_argument('--save_path', '-sp', type=Path, default=Path.cwd())
    argparser.add_argument('--load_file', '-lf', type=Path, default=Path.cwd())
    argparser.add_argument(
        '--no_overwrite', '-eo',
        type=bool, action='store_true'
    )
    argparser.add_argument(
        '--batch_size', '-bs',
        type=int, default=10
    )
    sys.exit(main(argparser.parse_args(sys.argv[1:])))
