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


def analize_json(
    value: Any,
    type_dict: dict[str, Any] | None = None
):
    if type_dict is None:
        type_dict = {}
    if isinstance(value, list):
        return [analize_json(value, type_dict) for value in value]
    elif isinstance(value, tuple):
        return tuple(analize_json(value, type_dict) for value in value)
    elif isinstance(value, dict):
        return {
            name: analize_json(value, type_dict)
            for name, value in value.items()
        }
    elif isinstance(value, str) and value in type_dict:
        return type_dict[value]
    else:
        return value


def analize_layers_param(
    value: Any,
    type_dict: dict[str, Any] | None = None
) -> dict[int, Any]:
    """
    A
    """
    if isinstance(value, dict):
        return {
            int(k): analize_json(value, type_dict)
            for k, value in value.items()
        }
    else:
        return analize_json(value, type_dict)

def tuple_encoder(object_dict: dict):
    if len(object_dict) == 1 and 'tuple' in object_dict:
        return tuple(object_dict['tuple'])
    return object_dict

def read_nn_json(file_path: Path, type_dict: dict | None = None):
    if type_dict is None:
        type_dict = nn_type_dict
    with file_path.open('r', encoding='utf-8') as fp:
        json_dict = json.load(fp=fp, object_hook=tuple_encoder)
    result_dict = {}
    for key, value in json_dict.items():
        if key == 'layers_params':
            for param, layer_values in json_dict['layers_params'].items():
                result_dict[param] = analize_layers_param(
                    value=layer_values,
                    type_dict=type_dict
                )
        else:
            result_dict[key] = analize_json(value=value, type_dict=type_dict)
    return result_dict
