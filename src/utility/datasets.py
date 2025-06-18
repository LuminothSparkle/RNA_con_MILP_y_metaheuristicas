"""
Modulo que contiene las clases de los datasets utilizados
"""

import re
from pathlib import Path
from pandas import Index, read_csv

from ucimlrepo import fetch_ucirepo, dotdict

from src.utility.nn.crossvalidation import CrossvalidationTensorDataset

class BCWDataset(CrossvalidationTensorDataset) :
    """
    Clase que implementa el dataset de breast cancer winsconsin
    """
    uci_dataset : dotdict

    def __init__(self, cv_data : dict | None = None, BCW_file_path : Path | None = None) -> None:
        if BCW_file_path is not None :
            labels = Index([
                'ID',
                *[
                    f'{s}{idx}' for idx in range(1,4)
                    for s in [
                        'radius', 'texture', 'perimeter','area', 'smoothness',
                        'compactness','concavity','concave_points','symmetry','fractal_dimension'
                    ]
                ],
                'Diagnosis'
            ])
            dataframe = read_csv(
                BCW_file_path, header = None, index_col = 0, names = labels
            )
        else :
            self.uci_dataset = fetch_ucirepo(id = 17)
            dataframe = self.uci_dataset.data.original # type: ignore
            labels = dataframe.columns
        labels_dict = {}
        for label_type,re_labels in cv_data['labels'].items() :
            labels_dict[label_type] = [
                label for label in labels
                if any(
                    re.match( pattern = re_label, string = label)
                    for re_label in re_labels
                )
            ]
        super().__init__(
            dataframe = dataframe,
            labels = labels_dict,
            data_augment = cv_data['data augment']
        )

class KardexDataset(CrossvalidationTensorDataset) :
    """
    Clase que implementa el dataset del kardex
    """

    def __init__(self, kardex_csv_path : Path, cv_data : dict | None = None) -> None:
        dataframe = read_csv(
            filepath_or_buffer = Path(kardex_csv_path),
            header = 0, index_col = 0
        )
        labels = dataframe.columns
        labels_dict = {}
        for label_type,re_labels in cv_data['labels'].items() :
            labels_dict[label_type] = [
                label for label in labels
                if any(
                    re.match( pattern = re_label, string = label)
                    for re_label in re_labels
                )
            ]
        super().__init__(
            dataframe = dataframe,
            labels = labels_dict,
            data_augment = cv_data['data augment']
        )
