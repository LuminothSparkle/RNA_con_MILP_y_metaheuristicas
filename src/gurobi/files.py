"""
Archivo con las funciones necesarias para interactuar
con elentrenamiento en C++ con gurobi
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
from argparse import ArgumentParser
import numpy
from numpy import ndarray
import pandas
from pandas import DataFrame, Index, read_csv
from torch.nn import Hardtanh, PReLU, Dropout, LeakyReLU
from crossvalidation.cvdir import safe_suffix
from src.utility.nn.lineal import LinealNN
from src.utility.io.model import load_model, save_model


def tensor_list_to_dataframe(tensor_list: list[ndarray | None]):
    """
    Generacion de un dataframe de una lista de tensores
    """
    dim_rows = []
    tensor_rows = []
    for tensor in tensor_list:
        if tensor is not None:
            dim_rows += [DataFrame(
                data={
                    f'd_{dim}': [shape]
                    for dim, shape in enumerate(tensor.shape)
                },
                dtype='Int64'
            )]
            tensor_rows += [DataFrame(data=[tensor.T.flatten()])]
        else:
            dim_rows += [DataFrame()]
            tensor_rows += [DataFrame()]
    dim_dataframe = pandas.concat(
        dim_rows, axis='index',
        ignore_index=True, join='outer'
    ).fillna(0)
    tensor_dataframe = pandas.concat(
        tensor_rows, axis='index',
        ignore_index=True, join='outer'
    )
    return pandas.concat([
        dim_dataframe, tensor_dataframe], axis='columns',
        ignore_index=False, join='outer'
    )


def dataframe_to_tensor_list(dataframe: DataFrame):
    """
    Generacion de una lista de tensor a partir de un dataframe
    """
    dim = len(dataframe.columns.str.startswith('d_'))
    dim_index = Index([f'd_{d}' for d in range(dim)])
    data_index = dataframe.columns.drop(dim_index)
    tensor_list = []
    for dim, data in zip(
        dataframe[dim_index].to_numpy(),
        dataframe[data_index].to_numpy()
    ):
        tensor_list += [data.view(dim).T]
    return tensor_list


def read_sol_file(file_path: Path) -> list[ndarray]:
    """
    Lectura del archivo .sol devuelto por gurobi
    """
    sol_dataframe = read_csv(
        filepath_or_buffer=file_path, sep=' ',
        index_col=0, skiprows=1, header=None
    )
    w_var_dataframe = sol_dataframe.loc[
        sol_dataframe.index.str.startswith('w_'), :
    ]
    layers = 1
    for name in w_var_dataframe.index:
        _, k, = name.split('_')
        k = int(k)
        layers = max(k + 1, layers)
    dim_list = [[1, 1] for _ in range(layers)]
    for name in w_var_dataframe.index:
        _, k, i, j, = name.split('_')
        i, j, k = int(k), int(j), int(k)
        dim_list[k][0] = max(dim_list[k][0], i + 1)
        dim_list[k][1] = max(dim_list[k][1], j + 1)
    w_list = [numpy.zeros(dim) for dim in dim_list]
    for name, value in w_var_dataframe.itertuples(index=True):
        _, k, i, j, = name.split('_')
        i, j, k = int(k), int(j), int(k)
        w_list[k][i, j] = float(value)
    return w_list


def read_files(
    dir_path: Path, read_name: str = '',
    not_exists_ok: bool = True
):
    """
    Temp
    """
    gurobi_path = dir_path / 'gurobi'
    model_path = dir_path / 'model'
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_model = executor.submit(
            load_model,
            dir_path=model_path,
            name=read_name,
            not_exists_ok=not_exists_ok
        )
        future_sol = executor.submit(
            read_sol_file,
            gurobi_path / f'{read_name}.sol'
        )
        module = future_model.result()
        weights = future_sol.result()
    if isinstance(module, LinealNN):
        module.set_weights(weights)
    return module


def write_files(
    module: LinealNN,
    dir_path: Path, save_name: str = '',
    zero_tolerance: float = 0.00001, min_bits: int = 8,
    exists_ok=True
):
    """
    Temp
    """
    gurobi_path = dir_path / 'gurobi'
    gurobi_path.mkdir(parents=True, exist_ok=True)
    files_suffix = ['bias', 'exp', 'bits', 'mask',
                    'arch', 'lReLU', 'cls_tgt', 'reg_tgt', 'ftr']
    files_path_dict = {
        file_type: gurobi_path / f'{safe_suffix(save_name, file_type)}.csv'
        for file_type in files_suffix
    }
    layers, act_layers = module.layers, module.activation_layers
    act_func = module.hyperparams['activation function']
    weights = module.get_weights()
    mantissa, exponent = zip(*(numpy.frexp(weight.T) for weight in weights))
    exponent = list(exponent)
    bits = [
        numpy.minimum(
            numpy.abs(numpy.floor_divide(1, numpy.log2(numpy.abs(m)))),
            numpy.full_like(m, min_bits)
        ).astype(int) for m in mantissa
    ]
    connections = [
        (numpy.absolute(weight).T > zero_tolerance).astype(int)
        for weight in weights
    ]
    leaky_relu = [
        act_layers[k].weight.cpu().detach().numpy()  # type: ignore
        if k in act_layers and isinstance(act_layers[k], PReLU)
        else numpy.full(
            (module.capacity[int(k) + 1],),
            act_layers[k].negative_slope
        )
        if k in act_layers and isinstance(act_layers[k], LeakyReLU)
        else numpy.full((module.capacity[int(k) + 1],), 0.25)
        for k in (str(k) for k in range(layers))
    ]
    assert exists_ok or not files_path_dict['lReLU'].exists(), (
        f"El archivo {files_path_dict['lReLU']} ya existe"
    )
    tensor_list_to_dataframe(leaky_relu).to_csv(  # type: ignore
        files_path_dict['lReLU'], header=True, index=False
    )
    assert exists_ok or not files_path_dict['mask'].exists(), (
        f"El archivo {files_path_dict['mask']} ya existe"
    )
    tensor_list_to_dataframe(connections).astype('Int64').to_csv(
        files_path_dict['mask'], header=True, index=False
    )
    assert exists_ok or not files_path_dict['bits'].exists(), (
        f"El archivo {files_path_dict['bits']} ya existe"
    )
    tensor_list_to_dataframe(bits).astype('Int64').to_csv(
        files_path_dict['bits'], header=True, index=False
    )
    assert exists_ok or not files_path_dict['exp'].exists(), (
        f"El archivo {files_path_dict['exp']} ya existe"
    )
    tensor_list_to_dataframe(exponent).astype('Int64').to_csv(
        files_path_dict['exp'], header=True, index=False
    )
    arch = {}
    arch['cap'] = module.capacity
    arch['act'] = [
        act_func[k][0] if k in act_func else 'None'
        for k in range(layers + 1)
    ]
    arch['dp'] = [
        act_layers[k].p
        if k in act_layers and isinstance(act_layers[k], Dropout)
        else None
        for k in (
            str(k) for k in range(layers + 1)
        )
    ]
    arch['ht_min'] = [
        act_layers[k].min_val
        if k in act_layers and isinstance(act_layers[k], Hardtanh)
        else -1
        for k in (
            str(k) for k in range(layers + 1)
        )
    ]
    arch['ht_max'] = [
        act_layers[k].max_val
        if k in act_layers and isinstance(act_layers[k], Hardtanh)
        else 1
        for k in (
            str(k) for k in range(layers + 1)
        )
    ]
    l1w = module.hyperparams['l1 weight regularization']
    arch['l1w'] = [
        l1w[k] if k in l1w else None
        for k in range(layers + 1)
    ]
    l1a = module.hyperparams['l1 activation regularization']
    arch['l1a'] = [
        l1a[k] if k in l1a else None
        for k in range(layers + 1)
    ]
    bias = module.hyperparams['bias init']
    arch['bias'] = [
        bias[k] if k in bias else 1
        for k in range(layers + 1)
    ]
    assert exists_ok or not files_path_dict['arch'].exists(), (
        f"El archivo {files_path_dict['arch']} ya existe"
    )
    DataFrame(arch).to_csv(
        files_path_dict['arch'], header=True, index=False
    )


def main(args: argparse.Namespace):
    """
    Funcion main para generar la base de datos en archivo csv del archivo
    json del kardex anonimizado
    """
    load_path = Path(args.load_path)
    save_path = Path(args.save_path)
    if not save_path.exists():
        print(f'{save_path} doesn\'t exists')
        return None
    elif not save_path.is_dir():
        print(f'Cannot access {save_path} or isn\'t a directory')
        return None
    if not load_path.exists():
        print(f'{load_path} doesn\'t exists')
        return None
    elif not load_path.is_dir():
        print(f'Cannot access {load_path} or isn\'t a directory')
        return None
    file_prefix = args.case_name
    if not args.from_gurobi:
        models_path = load_path / 'models'
        gurobi_path = save_path / 'gurobi'
        if not models_path.exists():
            print(f'{models_path} doesn\'t exists')
            return None
        elif not models_path.is_dir():
            print(f'Cannot access {models_path} or isn\'t a directory')
            return None
        module = load_model(
            models_path,
            name=file_prefix,
            not_exists_ok=False
        )
        if isinstance(module, LinealNN):
            write_files(
                module=module,
                dir_path=save_path, save_name=file_prefix,
                zero_tolerance=args.zero_tolerance,
                min_bits=args.min_bits,
                exists_ok=not args.no_overwrite
            )
    else:
        models_path = save_path / 'models'
        gurobi_path = load_path / 'gurobi'
        if not gurobi_path.exists():
            print(f'{gurobi_path} doesn\'t exists')
            return None
        elif not gurobi_path.is_dir():
            print(f'Cannot access {gurobi_path} or isn\'t a directory')
            return None
        module = read_files(load_path, file_prefix)
        models_path.mkdir(parents=True, exist_ok=True)
        model_name = safe_suffix(file_prefix, 'gb')
        if isinstance(module, LinealNN):
            save_model(
                module=module,
                dir_path=models_path,
                name=model_name,
                exists_ok=not args.no_overwrite
            )


if __name__ == '__main__':
    import sys
    argparser = ArgumentParser()
    argparser.add_argument(
        '--from_gurobi', '-fg',
        type=bool, action='store_true'
    )
    argparser.add_argument('--save_path', '-sp', type=Path, default=Path.cwd())
    argparser.add_argument('--load_path', '-lp', type=Path, default=Path.cwd())
    argparser.add_argument(
        '--zero_tolerance', '-zt',
        type=float, default=0.000001
    )
    argparser.add_argument('--min_bits', '-mb', type=int, default=8)
    argparser.add_argument(
        '--no_overwrite', '-eo',
        type=bool, action='store_true'
    )
    argparser.add_argument('--case_name', '-cn', type=str, default='')
    sys.exit(main(argparser.parse_args(sys.argv[1:])))
