"""
A
"""
from collections.abc import Iterable
from pathlib import Path
import torch
from torch.utils.data import Subset, Dataset
from src.utility.nn.cvdataset import CrossvalidationDataset


def save_dataset_csv(
    dataset: CrossvalidationDataset,
    file_path: Path, subset: str | Iterable[int] | None = None,
    label_type: str | Iterable[str] | None = None,
    raw: bool = False, exists_ok: bool = True
):
    """
    A
    """
    assert exists_ok or not file_path.exists(), (
        f"El archivo {file_path} ya existe"
    )
    dataframe = dataset.to_dataframe(subset, label_type, raw)
    assert dataframe is not None, (
        "No se pudo crear el archivo"
    )
    dataframe.to_csv(
        file_path,
        index_label='ID',
        index=True,
        header=True,
        encoding='utf-8'
    )


def save_pytorch_dataset(
    dataset: Subset | Dataset | CrossvalidationDataset,
    file_path: Path,
    exists_ok: bool = True
):
    """
    A
    """
    assert exists_ok or not file_path.exists(), (
        f"El archivo {file_path} ya existe"
    )
    torch.save(
        obj=dataset,
        f=file_path
    )


def load_pytorch_dataset(
    file_path: Path,
    not_exists_ok: bool = True
):
    """
    A
    """
    assert not_exists_ok or file_path.exists(), (
        f"El archivo {file_path} no existe"
    )
    return torch.load(
        f=file_path,
        map_location=torch.get_default_device(),
        weights_only=False
    )
