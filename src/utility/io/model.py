"""
A
"""
from copy import deepcopy
from typing import Any
from pathlib import Path
import pickle
import torch
from src.utility.nn.lineal import LinealNN, set_defaults


def save_model(
    module: LinealNN, dir_path: Path,
    name: str = '', exists_ok: bool = True
):
    """_summary_

    Args:
        module (LinealNN): _description_
        dir_path (Path): _description_
        name (str, optional): _description_. Defaults to ''.
        exists_ok (bool, optional): _description_. Defaults to True.
    """
    set_defaults()
    save_torch_model(module, dir_path / f'{name}.pt', exists_ok)
    save_torch_model_state(module, dir_path / f'{name}_state.pt', exists_ok)
    save_onnx_model(module, dir_path / f'{name}.onnx', exists_ok)
    save_train_onnx_model(module, dir_path / f'{name}_train.onnx', exists_ok)
    save_traced_model(module, dir_path / f'{name}_traced.pt', exists_ok)


def load_model(dir_path: Path, name: str = '', not_exists_ok: bool = True):
    """
    A
    """
    set_defaults()
    model = None
    try:
        model = load_torch_model(dir_path / f'{name}.pt', False)
        assert isinstance(model, LinealNN), (
            f"No se leyo correctamente {name}.pt"
        )
        return model
    except AssertionError as ae:
        print(ae)
    try:
        model = load_torch_model_state(dir_path / f'{name}_state.pt', False)
        assert isinstance(model, LinealNN), (
            f"No se leyo correctamente {name}_state.pt"
        )
        return model
    except AssertionError as ae:
        print(ae)
    assert not_exists_ok or not isinstance(model, LinealNN), (
        f"No existe algun archivo compatible con el nombre {name}"
    )
    return model


def save_torch_model(
    module: LinealNN, file_path: Path,
    exists_ok: bool = True
):
    """_summary_

    Args:
        module (LinealNN): _description_
        file_path (Path): _description_
        exists_ok (bool, optional): _description_. Defaults to True.
    """
    set_defaults()
    assert exists_ok or not file_path.exists(
    ), f'El archivo {file_path} ya existe'
    torch.save(module, file_path)


def load_torch_model(file_path: Path, not_exists_ok: bool = False):
    """_summary_

    Args:
        module (LinealNN): _description_
        file_path (Path): _description_
        exists_ok (bool, optional): _description_. Defaults to True.
    """
    set_defaults()
    assert not_exists_ok or file_path.exists(), (
        f"El archivo {file_path} no existe"
    )
    if file_path.exists():
        return torch.load(
            f=file_path,
            map_location=torch.get_default_device(),
            weights_only=False
        )
    return None


def save_torch_model_state(
    module: LinealNN, file_path: Path,
    exists_ok: bool = True
):
    """_summary_

    Args:
        module (LinealNN): _description_
        file_path (Path): _description_
        exists_ok (bool, optional): _description_. Defaults to True.
    """
    set_defaults()
    assert exists_ok or not file_path.exists(
    ), f'El archivo {file_path} ya existe'
    torch.save(module.state_dict(), file_path)


def load_torch_model_state(file_path: Path, not_exists_ok: bool = False):
    """_summary_

    Args:
        module (LinealNN): _description_
        file_path (Path): _description_
        exists_ok (bool, optional): _description_. Defaults to True.
    """
    set_defaults()
    assert not_exists_ok or file_path.exists(), (
        f"El archivo {file_path} no existe"
    )
    if file_path.exists():
        state = torch.load(
            f=file_path,
            map_location=torch.get_default_device(),
            weights_only=False
        )
        model = LinealNN()
        model.load_state_dict(state, assign=True, strict=False)
        return model
    return None


def save_train_onnx_model(
    module: LinealNN, file_path: Path,
    exists_ok: bool = True
):
    """
    A
    """
    set_defaults()
    module.train()
    save_onnx_model(module, file_path, exists_ok)
    module.eval()


def save_onnx_model(module: LinealNN, file_path: Path, exists_ok: bool = True):
    """_summary_

    Args:
        module (LinealNN): _description_
        file_path (Path): _description_
        exists_ok (bool, optional): _description_. Defaults to True.
    """
    assert exists_ok or not file_path.exists(
    ), f'El archivo {file_path} ya existe'
    torch.set_default_device('cpu')
    torch.set_default_dtype(torch.double)
    cpu_module = LinealNN()
    cpu_module.load_state_dict(deepcopy(module.state_dict()), assign=True, strict=False)
    cpu_module.to(device='cpu', dtype=torch.double, non_blocking=True)
    torch.onnx.export(
        model=cpu_module,
        f=file_path,
        args=(torch.ones((1, module.capacity[0]), device='cpu', dtype=torch.double),),
        input_names=['Features'],
        output_names=['Preprobabilities'],
        export_params=True
    )


def save_script_model(
    module: LinealNN, file_path: Path,
    exists_ok: bool = True
):
    """_summary_

    Args:
        module (LinealNN): _description_
        file_path (Path): _description_
        exists_ok (bool, optional): _description_. Defaults to True.
    """
    set_defaults()
    assert exists_ok or not file_path.exists(
    ), f'El archivo {file_path} ya existe'
    torch.jit.script(
        obj=module,
        example_inputs=[(torch.ones((1, module.capacity[0])),)],
        optimize=True
    ).save(file_path)


def save_traced_model(
    module: LinealNN, file_path: Path,
    exists_ok: bool = True
):
    """_summary_

    Args:
        module (LinealNN): _description_
        file_path (Path): _description_
        exists_ok (bool, optional): _description_. Defaults to True.
    """
    set_defaults()
    assert exists_ok or not file_path.exists(
    ), f'El archivo {file_path} ya existe'
    torch.jit.save(
        torch.jit.trace_module(
            mod=module,
            inputs={
                'forward': (torch.ones(1, module.capacity[0])),
            }
        ),
        file_path
    )


def load_script_model(file_path: Path, not_exists_ok: bool = False):
    """_summary_

    Args:
        file_path (Path): _description_

    Returns:
        _type_: _description_
    """
    set_defaults()
    assert not_exists_ok or file_path.exists(), (
        f"El archivo {file_path} no existe"
    )
    if file_path.exists():
        return torch.jit.load(file_path)
    return None


def save_object(obj: Any, file_path: Path):
    """
    A
    """
    with file_path.open('wb') as fo:
        pickle.dump(obj, fo, -1)


def read_object(file_path: Path):
    """
    A
    """
    with file_path.open('rb') as fo:
        obj = pickle.load(fo)
    return obj
