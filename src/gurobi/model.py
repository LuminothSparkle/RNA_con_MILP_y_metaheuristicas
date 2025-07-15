"""
Archivo con las funciones necesarias para interactuar
con el entrenamiento en C++ con gurobi
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
from argparse import ArgumentParser
import numpy
from numpy import ndarray
import pandas
from pandas import DataFrame, Index, read_csv
from torch.nn import Hardtanh, PReLU, LeakyReLU
from crossvalidation.files import safe_suffix
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
    layers = 0
    for name in w_var_dataframe.index:
        _, k, _, _ = name.split('_')
        layers = max(int(k) + 1, layers)
    dim_list = [[0, 0] for _ in range(layers)]
    for name in w_var_dataframe.index:
        _, k, i, j, = name.split('_')
        i, j, k = int(i), int(j), int(k)
        dim_list[k][0] = max(dim_list[k][0], i + 1)
        dim_list[k][1] = max(dim_list[k][1], j + 1)
    w_list = [numpy.zeros(dim) for dim in dim_list]
    for name, value in w_var_dataframe.itertuples(index=True):
        _, k, i, j, = name.split('_')
        i, j, k = int(i), int(j), int(k)
        w_list[k][i, j] = float(value)
    return [w.T for w in w_list]


def read_files(
    models_path: Path, gurobi_path: Path,
    read_name: str = '',
    case_index: str = '',
    not_exists_ok: bool = True
):
    """
    Temp
    """
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_model = executor.submit(
            load_model,
            dir_path=models_path / safe_suffix('model', case_index),
            name=read_name,
            not_exists_ok=not_exists_ok
        )
        future_sol = executor.submit(
            read_sol_file,
            gurobi_path / safe_suffix('model', case_index)
            / f'{safe_suffix(read_name, "gurobi")}.sol'
        )
        module = future_model.result()
        weights = future_sol.result()
        print([w.shape for w in weights])
    if isinstance(module, LinealNN):
        module.set_weights(weights)
    return module


def write_files(
    module: LinealNN,
    dir_path: Path, save_name: str = '',
    zero_tolerance: float = 0.00001,
    exists_ok=True
):
    """
    Temp
    """
    dir_path.mkdir(parents=True, exist_ok=True)
    files_suffix = [
        'bias', 'exp', 'bits', 'mask', 'arch',
        'lrelu', 'cls_tgt', 'reg_tgt', 'ftr', 'init'
    ]
    files_path_dict = {
        file_type: dir_path / f'{safe_suffix(save_name, file_type)}.csv'
        for file_type in files_suffix
    }
    layers, act_layers = module.layers, module.activation_layers
    masks = [mask.cpu().detach().numpy() for mask in module.masks]
    weights = module.get_weights()
    mantissa, exponent = zip(*(numpy.frexp(weight.T) for weight in weights))
    mantissa = list(mantissa)
    exponent = list(exponent)
    bits = []
    for m in mantissa:
        mbit = numpy.full_like(m, 0)
        m = m + (m < 0)
        while (m > zero_tolerance).any():
            mbit[m > zero_tolerance] += 1
            m, _ = numpy.modf(2 * m)
        bits += [mbit]
    connections = [
        (mask.T * (numpy.abs(weight).T > zero_tolerance)).astype(int)
        for weight, mask in zip(weights, masks)
    ]
    leaky_relu = [
        act_layers[k].weight.cpu().detach().numpy()  # type: ignore
        if isinstance(act_layers[k], PReLU)
        else numpy.full(
            (module.capacity[int(k) + 1],),
            act_layers[k].negative_slope
        )
        if isinstance(act_layers[k], LeakyReLU)
        else numpy.full((module.capacity[int(k) + 1],), 0.25)
        for k in range(layers)
    ]
    assert exists_ok or not files_path_dict['init'].exists(), (
        f"El archivo {files_path_dict['init']} ya existe"
    )
    tensor_list_to_dataframe(weights).to_csv(  # type: ignore
        files_path_dict['init'], header=True, index=False
    )
    assert exists_ok or not files_path_dict['lrelu'].exists(), (
        f"El archivo {files_path_dict['lrelu']} ya existe"
    )
    tensor_list_to_dataframe(leaky_relu).to_csv(  # type: ignore
        files_path_dict['lrelu'], header=True, index=False
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
        *(
            act[0]
            for act in module.activation
        ),
        None
    ]
    arch['dp'] = [*module.dropout, None]
    arch['ht_min'] = [
        *(
            act_layers[k].min_val
            if isinstance(act_layers[k], Hardtanh)
            else -1
            for k in range(layers)
        ),
        None
    ]
    arch['ht_max'] = [
        *(
            act_layers[k].max_val
            if isinstance(act_layers[k], Hardtanh)
            else 1
            for k in range(layers)
        ),
        None
    ]
    arch['l1w'] = [*module.l1_weight, None]
    arch['l1a'] = [*module.l1_activation, None]
    arch['l2w'] = [*module.l2_weight, None]
    arch['l2a'] = [*module.l2_activation, None]
    arch['bias'] = [*module.bias, None]
    arch['cdp'] = [*module.connection_dropout, None]
    assert exists_ok or not files_path_dict['arch'].exists(), (
        f"El archivo {files_path_dict['arch']} ya existe"
    )
    DataFrame(arch).to_csv(
        files_path_dict['arch'], header=True, index=False
    )


def main(args: argparse.Namespace):
    """
    A
    """
    if not args.gen_model:
        model_path = args.load_path / safe_suffix('model', args.case_index)
        module = load_model(
            model_path,
            name=args.load_name,
            not_exists_ok=False
        )
        save_path = args.save_path / safe_suffix('model', args.case_index)
        if isinstance(module, LinealNN):
            write_files(
                module=module,
                dir_path=save_path,
                save_name=args.save_name,
                zero_tolerance=args.zero_tolerance,
                exists_ok=not args.no_overwrite
            )
    else:
        module = read_files(
            models_path=args.load_path / 'models',
            gurobi_path=args.load_path / 'gurobi',
            read_name=args.load_name,
            case_index=args.case_index
        )
        if isinstance(module, LinealNN):
            model_path = (
                args.save_path
                / safe_suffix('model', args.case_index)
            )
            model_path.mkdir(parents=True, exist_ok=True)
            save_model(
                module=module,
                dir_path=model_path,
                name=args.load_name,
                exists_ok=not args.no_overwrite
            )


if __name__ == '__main__':
    import sys
    argparser = ArgumentParser()
    argparser.add_argument(
        '--gen_model', '-gm',
        action='store_true'
    )
    argparser.add_argument('--save_path', '-sp', type=Path, default=Path.cwd())
    argparser.add_argument('--load_path', '-lp', type=Path, default=Path.cwd())
    argparser.add_argument(
        '--zero_tolerance', '-zt',
        type=float, default=0.000001
    )
    argparser.add_argument(
        '--no_overwrite', '-eo',
        action='store_true'
    )
    argparser.add_argument('--load_name', '-ln', type=str, default='')
    argparser.add_argument('--save_name', '-sn', type=str, default='')
    argparser.add_argument('--case_index', '-ci', type=str, default='')
    sys.exit(main(argparser.parse_args(sys.argv[1:])))
