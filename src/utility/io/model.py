"""
A
"""
from pathlib import Path
import torch
import utility.nn.torchdefault as torchdefault
from utility.nn.lineal import LinealNN


def save_model(module: LinealNN, dir_path: Path, name: str = ''):
    """_summary_

    Args:
        module (LinealNN): _description_
        dir_path (Path): _description_
        name (str, optional): _description_. Defaults to ''.
        exists_ok (bool, optional): _description_. Defaults to True.
    """
    torchdefault.set_defaults()
    save_onnx_model(module, dir_path / f'{name}.onnx')
    save_train_onnx_model(module, dir_path / f'{name}_train.onnx')
    save_traced_model(module, dir_path / f'{name}_traced.pt')


def save_train_onnx_model(module: LinealNN, file_path: Path):
    """
    A
    """
    module.train()
    save_onnx_model(module, file_path)
    module.eval()


def save_onnx_model(module: LinealNN, file_path: Path):
    """
    A
    """
    torch.set_default_device('cpu')
    torch.set_default_dtype(torch.double)
    cpu_module = module.copy()
    cpu_module.to(device='cpu', dtype=torch.double, non_blocking=True)
    torch.onnx.export(
        model=cpu_module,
        f=file_path,
        args=(
            torch.ones(
                (1, module.capacity[0]),
                device='cpu', dtype=torch.double
            ),
        ),
        input_names=['Features'],
        output_names=['Values'],
        export_params=True
    )
    torchdefault.set_defaults()


def save_script_model(module: LinealNN, file_path: Path):
    """_summary_

    Args:
        module (LinealNN): _description_
        file_path (Path): _description_
        exists_ok (bool, optional): _description_. Defaults to True.
    """
    torchdefault.set_defaults()
    torch.jit.script(
        obj=module,
        example_inputs=[(torch.ones((1, module.capacity[0])),)],
        optimize=True
    ).save(file_path)


def save_traced_model(module: LinealNN, file_path: Path):
    """_summary_

    Args:
        module (LinealNN): _description_
        file_path (Path): _description_
        exists_ok (bool, optional): _description_. Defaults to True.
    """
    torchdefault.set_defaults()
    torch.jit.save(
        torch.jit.trace_module(
            mod=module,
            inputs={
                'forward': (torch.ones(1, module.capacity[0])),
            }
        ),
        file_path
    )


def load_script_model(file_path: Path):
    """_summary_

    Args:
        file_path (Path): _description_

    Returns:
        _type_: _description_
    """
    torchdefault.set_defaults()
    return torch.jit.load(file_path, map_location=torch.get_default_device())