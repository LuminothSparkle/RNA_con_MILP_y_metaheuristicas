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
from utility.io.dataset import load_pytorch_dataset


def generate_csv(
    load_file: Path,
    save_path: Path,
    exists_ok: bool = True
):
    """
    A
    """
    save_path.mkdir(parents=True, exist_ok=True)
    dataset = load_pytorch_dataset(file_path=load_file)
    ftr_trai_dat, reg_trai_dat, cls_trai_dat = dataset.dataset.to_tensor_dataframes(dataset.dataset.train_dataframe, True, True)
    ftr_test_dat, reg_test_dat, cls_test_dat = dataset.dataset.to_tensor_dataframes(dataset.dataset.test_dataframe, False, False)
    ftr_vali_dat, reg_vali_dat, cls_vali_dat = dataset.dataset.to_tensor_dataframes(dataset.dataset.validation_dataframe, False, False)
    file_path = save_path / f'{load_file.stem}_train_ftr.csv'
    assert exists_ok or not file_path.exists(), (
        f"El archivo {file_path} ya existe"
    )
    ftr_trai_dat.to_csv(
        file_path,
        header=True,
        index=True,
        index_label='ID'
    )
    file_path = save_path / f'{load_file.stem}_test_ftr.csv'
    assert exists_ok or not file_path.exists(), (
        f"El archivo {file_path} ya existe"
    )
    ftr_test_dat.to_csv(
        file_path,
        header=True,
        index=True,
        index_label='ID'
    )
    file_path = save_path / f'{load_file.stem}_validation_ftr.csv'
    assert exists_ok or not file_path.exists(), (
        f"El archivo {file_path} ya existe"
    )
    ftr_vali_dat.to_csv(
        file_path,
        header=True,
        index=True,
        index_label='ID'
    )
    file_path = save_path / f'{load_file.stem}_train_reg_tgt.csv'
    assert exists_ok or not file_path.exists(), (
        f"El archivo {file_path} ya existe"
    )
    reg_trai_dat.to_csv(
        file_path,
        header=True,
        index=True,
        index_label='ID'
    )
    file_path = save_path / f'{load_file.stem}_test_reg_tgt.csv'
    assert exists_ok or not file_path.exists(), (
        f"El archivo {file_path} ya existe"
    )
    reg_test_dat.to_csv(
        file_path,
        header=True,
        index=True,
        index_label='ID'
    )
    file_path = save_path / f'{load_file.stem}_validation_reg_tgt.csv'
    assert exists_ok or not file_path.exists(), (
        f"El archivo {file_path} ya existe"
    )
    reg_vali_dat.to_csv(
        file_path,
        header=True,
        index=True,
        index_label='ID'
    )
    for cls, dat in cls_trai_dat.items():
        file_path = save_path / f'{load_file.stem}_train_cls_tgt_{cls}.csv'
        assert exists_ok or not file_path.exists(), (
            f"El archivo {file_path} ya existe"
        )
        dat.to_csv(
            file_path,
            header=True,
            index=True,
            index_label='ID'
        )
    for cls, dat in cls_test_dat.items():
        file_path = save_path / f'{load_file.stem}_test_cls_tgt_{cls}.csv'
        assert exists_ok or not file_path.exists(), (
            f"El archivo {file_path} ya existe"
        )
        dat.to_csv(
            file_path,
            header=True,
            index=True,
            index_label='ID'
        )
    for cls, dat in cls_vali_dat.items():
        file_path = save_path / f'{load_file.stem}_validation_cls_tgt_{cls}.csv'
        assert exists_ok or not file_path.exists(), (
            f"El archivo {file_path} ya existe"
        )
        dat.to_csv(
            file_path,
            header=True,
            index=True,
            index_label='ID'
        )


def main(args: argparse.Namespace):
    """
    Funcion main para generar la base de datos en archivo csv del archivo
    json del kardex anonimizado
    """
    generate_csv(args.load_path, args.save_path, not args.no_overwrite)


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
        '--no_overwrite', '-eo',
        action='store_true'
    )
    sys.exit(main(argparser.parse_args(sys.argv[1:])))
