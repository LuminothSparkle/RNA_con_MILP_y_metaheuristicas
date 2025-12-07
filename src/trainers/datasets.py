"""
Modulo que contiene las clases de los datasets utilizados
"""
import re
from pathlib import Path
from pandas import Index, read_csv
from ucimlrepo import fetch_ucirepo
from utility.nn.dataset import CsvDataset


def bcw_dataset(
    file_path: Path | None = None,
    labels_regex: dict[str, list[str]] | None = None,
    **kwargs
):
    uci_dataset = None
    if labels_regex is None:
        labels_regex = {}
    if file_path is not None:
        labels_list = Index([
            'ID',
            'Diagnosis',
            *[
                f'{s}{idx}' for idx in range(1, 4)
                for s in [
                    'radius', 'texture', 'perimeter', 'area', 'smoothness',
                    'compactness', 'concavity', 'concave_points',
                    'symmetry', 'fractal_dimension'
                ]
            ]
        ])
        dataframe = read_csv(
            file_path, header=None, index_col=0,
            names=labels_list.to_list()
        )
    else:
        uci_dataset = fetch_ucirepo(id=17)
        dataframe = (
            uci_dataset.data.original  # type: ignore
        )
        labels_list = dataframe.columns
    labels = {}
    for label_type, re_labels in labels_regex.items():
        labels[label_type] = [
            label for label in labels_list
            if any(
                re.match(pattern=re_label, string=label)
                for re_label in re_labels
            )
        ]
    return CsvDataset.from_dataframe(dataframe=dataframe, labels=labels, **kwargs)

def kardex_dataset(
    file_path: Path,
    labels_regex: dict[str, list[str]] | None = None,
    **kwargs
):
    if labels_regex is None:
        labels_regex = {}
    dataframe = read_csv(
        filepath_or_buffer=Path(file_path),
        header=0, index_col=0
    )
    labels_list = dataframe.columns
    labels = {}
    for label_type, re_labels in labels_regex.items():
        labels[label_type] = [
            label for label in labels_list
            if any(
                re.match(pattern=re_label, string=label)
                for re_label in re_labels
            )
        ]
    return CsvDataset.from_dataframe(dataframe=dataframe, labels=labels, **kwargs)