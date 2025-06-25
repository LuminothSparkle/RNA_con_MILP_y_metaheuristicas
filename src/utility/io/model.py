"""
A
"""
from typing import Any
from pathlib import Path
import pickle
import torch
from src.utility.nn.lineal import LinealNN


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
    save_python_model(module, dir_path / f'{name}.pkl', exists_ok)
    save_torch_model(module, dir_path / f'{name}.pt', exists_ok)
    save_onnx_model(module, dir_path / f'{name}.onnx', exists_ok)
    save_script_model(module, dir_path / f'{name}_script.pt', exists_ok)


def load_model(dir_path: Path, name: str = '', not_exists_ok: bool = True):
    """
    A
    """
    model = None
    try:
        model = load_python_model(dir_path / f'{name}.pkl', False)
        assert isinstance(model, LinealNN)
    except AssertionError as ae:
        print(ae)
    try:
        model = load_torch_model(dir_path / f'{name}.pt', False)
        assert isinstance(model, LinealNN)
    except AssertionError as ae:
        print(ae)
    try:
        model = load_script_model(dir_path / f'{name}_script.pt', False)
        assert isinstance(model, LinealNN)
    except AssertionError as ae:
        print(ae)
    assert not_exists_ok or not isinstance(model, LinealNN), (
        f"No existe algun archivo compatible con el nombre {name}"
    )
    return model


def save_python_model(
    module: LinealNN, file_path: Path,
    exists_ok: bool = True
):
    """_summary_

    Args:
        module (LinealNN): _description_
        file_path (Path): _description_
        exists_ok (bool, optional): _description_. Defaults to True.
    """
    assert exists_ok or not file_path.exists(), (
        f"El archivo {file_path} ya existe"
    )
    save_object(module, file_path)


def load_python_model(file_path: Path, not_exists_ok: bool = False):
    """_summary_

    Args:
        module (LinealNN): _description_
        file_path (Path): _description_
        exists_ok (bool, optional): _description_. Defaults to True.
    """
    assert not_exists_ok or file_path.exists(), (
        f"El archivo {file_path} no existe"
    )
    if file_path.exists():
        return read_object(file_path)
    return None


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
    assert not_exists_ok or file_path.exists(), (
        f"El archivo {file_path} no existe"
    )
    if file_path.exists():
        return torch.load(file_path)
    return None


def save_onnx_model(module: LinealNN, file_path: Path, exists_ok: bool = True):
    """_summary_

    Args:
        module (LinealNN): _description_
        file_path (Path): _description_
        exists_ok (bool, optional): _description_. Defaults to True.
    """
    assert exists_ok or not file_path.exists(
    ), f'El archivo {file_path} ya existe'
    onnx_program = torch.onnx.export(
        model=module,
        args=(torch.ones((1, module.capacity[0])),),
        dynamo=True,
        export_params=True
    )
    assert onnx_program is not None
    onnx_program.optimize()
    onnx_program.save(file_path)


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
    assert exists_ok or not file_path.exists(
    ), f'El archivo {file_path} ya existe'
    torch.jit.script(torch.jit.trace(
        module, (torch.ones(1, module.capacity[0])))).save(file_path)


def load_script_model(file_path: Path, not_exists_ok: bool = False):
    """_summary_

    Args:
        file_path (Path): _description_

    Returns:
        _type_: _description_
    """
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
        pickle.dump(obj, fo, 5)


def read_object(file_path: Path):
    """
    A
    """
    with file_path.open('rb') as fo:
        obj = pickle.load(fo)
    return obj
