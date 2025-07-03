"""
34er7
"""
import json
from typing import Any
from pathlib import Path
from torch.optim import (
    Adadelta, Adafactor, Adagrad, Adam,
    Adamax, AdamW, ASGD, SGD, SparseAdam
)
from torch.nn.init import calculate_gain
from torch.optim.lr_scheduler import (
    LinearLR, SequentialLR, StepLR,
    CyclicLR, OneCycleLR, ConstantLR
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold, RepeatedKFold, KFold,
    StratifiedGroupKFold, StratifiedKFold, GroupKFold
)


nn_type_dict = {
    'RepeatedStratifiedKFold': RepeatedStratifiedKFold,
    'RepeatedKFold': RepeatedKFold,
    'KFold': KFold,
    'StratifiedGroupKFold': StratifiedGroupKFold,
    'StratifiedKFold': StratifiedKFold,
    'GroupKFold': GroupKFold,
    'LinearLR': LinearLR,
    'SequentialLR': SequentialLR,
    'StepLR': StepLR,
    'CyclicLR': CyclicLR,
    'OneCycleLR': OneCycleLR,
    'ConstantLR': ConstantLR,
    'Adadelta': Adadelta,
    'Adafactor': Adafactor,
    'Adagrad': Adagrad,
    'Adam': Adam,
    'Adamax': Adamax,
    'AdamW': AdamW,
    'ASGD': ASGD,
    'SGD': SGD,
    'SparseAdam': SparseAdam,
    'calculate_gain': calculate_gain
}


def gen_from_tuple(
    value: tuple[Any, list[Any], dict[str, Any]]
):
    """
    A
    """
    return value[0](*value[1], **value[2])


def analize_value_param(
    value: Any,
    type_dict: dict[str, Any] | None = None
) -> tuple | Any:
    """
    A
    """
    if type_dict is None:
        type_dict = {}
    if isinstance(value, dict) and 'tuple' in value:
        return tuple(analize_value_param(value['tuple'], type_dict))
    elif isinstance(value, list):
        return [analize_value_param(value, type_dict) for value in value]
    elif isinstance(value, dict):
        return {
            name: analize_value_param(value, type_dict)
            for name, value in value.items()
        }
    elif isinstance(value, str) and value in type_dict:
        return type_dict[value]
    else:
        return value


def analize_layers_param(
    value: Any, layers: int,
    type_dict: dict[str, Any] | None = None
) -> dict[int, Any]:
    """
    A
    """
    if type_dict is None:
        type_dict = {}
    if isinstance(value, list):
        return {
            k: analize_value_param(value, type_dict)
            for k, value in enumerate(value)
        }
    elif isinstance(value, dict) and 'tuple' not in value:
        return {
            int(k): analize_value_param(value, type_dict)
            for k, value in value.items()
        }
    else:
        return {k: analize_value_param(value) for k in range(layers)}


def read_arch_json(file_path: Path) -> dict:
    """
    A
    """
    arch = {}
    with file_path.open('rt', encoding='utf-8') as fp:
        raw = json.load(fp)
        arch['layers'] = len(raw['capacity']) + 1
        for param, values in raw.items():
            if param == 'layers params':
                for param, values in values.items():
                    arch[param] = analize_layers_param(
                        values,
                        arch['layers'],
                        nn_type_dict
                    )
            else:
                arch[param] = analize_value_param(values, nn_type_dict)
    return arch


def read_cv_json(file_path: Path) -> dict:
    """
    A
    """
    cv = {}
    with file_path.open('rt', encoding='utf-8') as fp:
        raw = json.load(fp)
        for param, values in raw.items():
            cv[param] = analize_value_param(values, nn_type_dict)
    return cv
