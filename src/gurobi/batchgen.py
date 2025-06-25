"""
A
"""
from pathlib import Path
import argparse
from argparse import ArgumentParser
from pandas import read_csv


def main(args: argparse.Namespace):
    """
    Funcion main para generar la base de datos en archivo csv del archivo
    json del kardex anonimizado
    """
    dataframe = read_csv(args.load_file, header=0, index_col=0)
    for batch, start in enumerate(range(0, len(dataframe), args.batch_size)):
        file_path = args.save_path / f'{args.load_file.stem()}_{batch}'
        assert not args.no_overwrite() or not file_path.exists(), (
            f"El archivo {file_path} ya existe"
        )
        dataframe[start:(start + args.batch_size), :].to_csv(
            path_or_buf=file_path,
            header=True,
            index=True,
            index_label='ID'
        )


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
