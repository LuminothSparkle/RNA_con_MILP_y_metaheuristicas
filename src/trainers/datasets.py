"""
Modulo que contiene las clases de los datasets utilizados
"""
import re
from pathlib import Path
from pandas import Index, read_csv
from sklearn.model_selection import BaseCrossValidator
import torch
from ucimlrepo import fetch_ucirepo, dotdict
from src.utility.nn.cvtensords import (
    CrossvalidationTensorDataset, CrossvalidationDataset
)
from src.utility.nn.cvdataset import crossvalidate


class BCWDataset(CrossvalidationTensorDataset):
    """
    Clase que implementa el dataset de breast cancer winsconsin
    """
    uci_dataset: dotdict

    def __init__(
        self, crossvalidator: BaseCrossValidator,
        BCW_file_path: Path | None = None,
        labels: dict[str, list[str]] | None = None,
        **kwargs
    ) -> None:
        kwargs['crossvalidator'] = crossvalidator
        if labels is None:
            labels = {}
        if BCW_file_path is not None:
            labels_list = Index([
                'ID',
                *[
                    f'{s}{idx}' for idx in range(1, 4)
                    for s in [
                        'radius', 'texture', 'perimeter', 'area', 'smoothness',
                        'compactness', 'concavity', 'concave_points',
                        'symmetry', 'fractal_dimension'
                    ]
                ],
                'Diagnosis'
            ])
            kwargs['dataframe'] = read_csv(
                BCW_file_path, header=None, index_col=0,
                names=labels_list.to_list()
            )
        else:
            self.uci_dataset = fetch_ucirepo(id=17)
            kwargs['dataframe'] = (
                self.uci_dataset.data.original  # type: ignore
            )
            labels_list = kwargs['dataframe'].columns
        kwargs['labels'] = {}
        for label_type, re_labels in labels.items():
            kwargs['labels'][label_type] = [
                label for label in labels_list
                if any(
                    re.match(pattern=re_label, string=label)
                    for re_label in re_labels
                )
            ]
        super().__init__(**kwargs)


class KardexDataset(CrossvalidationTensorDataset):
    """
    Clase que implementa el dataset del kardex
    """

    def __init__(
        self, kardex_csv_path: Path,
        crossvalidator: BaseCrossValidator,
        labels: dict[str, list[str]] | None = None,
        **kwargs
    ) -> None:
        if labels is None:
            labels = {}
        kwargs['dataframe'] = read_csv(
            filepath_or_buffer=Path(kardex_csv_path),
            header=0, index_col=0
        )
        labels_list = kwargs['dataframe'].columns
        kwargs['crossvalidator'] = crossvalidator
        kwargs['labels'] = {}
        for label_type, re_labels in labels.items():
            kwargs['labels'][label_type] = [
                label for label in labels_list
                if any(
                    re.match(pattern=re_label, string=label)
                    for re_label in re_labels
                )
            ]
        super().__init__(**kwargs)


def test_arch(
    dataset: CrossvalidationDataset, iterations: int,
    train_batches: int,
    **kwargs
):
    """
    A
    """
    torch.set_default_device('cpu')
    torch.set_default_dtype(torch.double)
    if torch.cuda.is_available():
        torch.set_default_device('cuda')
        torch.set_default_dtype(torch.double)
    return crossvalidate(
        dataset=dataset,
        iterations=iterations,
        train_batches=train_batches,
        **kwargs
    )
