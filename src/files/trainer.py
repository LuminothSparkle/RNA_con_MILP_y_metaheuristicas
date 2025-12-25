"""
Codigo que entrena la red neuronal de Breast Cancer Winsconsin Diagnostic
"""
import argparse
from argparse import ArgumentParser
from pathlib import Path
from time import sleep
from concurrent.futures import ThreadPoolExecutor
from utility.signal import InterruptHandler
import utility.nn.torchdefault as torchdefault
from utility.io.crossvalidation import load_trainer


def main(args: argparse.Namespace):
    """
    Funcion main para generar la base de datos en archivo csv del archivo
    json del kardex anonimizado
    """
    torchdefault.set_defaults()
    trainer = load_trainer(args.load_path)
    #df = comparative_dataframe([trainer], ['trainer'])
    #df.to_csv(path_or_buf=args.save_path)


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
    sys.exit(main(argparser.parse_args(sys.argv[1:])))
