"""
Archivo con las funciones necesarias para interactuar con el entrenamiento en C++ con gurobi
"""
from pathlib import Path

import argparse
from argparse import ArgumentParser

import numpy
from numpy import ndarray

import pandas
from pandas import DataFrame, Index, read_csv
from torch.nn import Hardtanh, PReLU, Dropout, LeakyReLU

from src.utility.nn.lineal import LinealNN
from src.utility.nn.crossvalidation import CrossvalidationDataset

from src.utility.files import write_mdb, read_mdb, save_module, safe_suffix

def tensor_list_to_dataframe(tensor_list : list[ndarray | None]) :
    """
    Generacion de un dataframe de una lista de tensores
    """
    dim_rows = []
    tensor_rows = []
    for tensor in tensor_list :
        if tensor is not None :
            dim_rows += [DataFrame(data = {f'd_{dim}' : shape
                                    for dim,shape in enumerate(tensor.shape)}, dtype = 'Int64')]
            tensor_rows += [DataFrame(data = [tensor.T.flatten()], dtype = tensor.dtype)]
        else :
            dim_rows += [DataFrame()]
            tensor_rows += [DataFrame()]
    dim_dataframe = pandas.concat(dim_rows, axis = 'index',
                                ignore_index = True, join = 'outer').fillna(0)
    tensor_dataframe = pandas.concat(tensor_rows, axis = 'index',
                                ignore_index = True, join = 'outer')
    return pandas.concat([dim_dataframe, tensor_dataframe], axis = 'columns',
                        ignore_index = False, join = 'fouter')

def dataframe_to_tensor_list(dataframe : DataFrame) :
    """
    Generacion de una lista de tensor a partir de un dataframe
    """
    dim = len(dataframe.columns.str.startswith('d_'))
    dim_index = Index([ f'd_{d}' for d in range(dim) ])
    data_index = dataframe.columns.drop(dim_index)
    tensor_list = []
    for dim,data in zip(dataframe[dim_index].to_numpy(), dataframe[data_index].to_numpy()) :
        tensor_list += [data.view(dim).T]
    return tensor_list

def read_sol_file(file_path : Path) -> list[ndarray] :
    """
    Lectura del archivo .sol devuelto por gurobi
    """
    sol_dataframe = read_csv(filepath_or_buffer = file_path, sep = ' ',
                            index_col = 0, skiprows = 1, header = None)
    w_var_dataframe = sol_dataframe.loc[sol_dataframe.index.str.startswith('w_'), :]
    layers = 1
    for name in w_var_dataframe.index :
        _,k, = name.split('_')
        k = int(k)
        layers = max(k + 1,layers)
    dim_list = [[1,1] for _ in range(layers)]
    for name in w_var_dataframe.index :
        _,k,i,j, = name.split('_')
        i, j, k = int(k), int(j), int(k)
        dim_list[k][0] = max(dim_list[k][0], i + 1)
        dim_list[k][1] = max(dim_list[k][1], j + 1)
    w_list = [ numpy.zeros(dim) for dim in dim_list]
    for name, value in w_var_dataframe.itertuples(index = True) :
        _,k,i,j, = name.split('_')
        i, j, k = int(k), int(j), int(k)
        w_list[k][i, j] = float(value)
    return w_list

def read_files(dir_path : Path, read_name : str = '') :
    """
    Temp
    """
    gurobi_path = dir_path / 'gurobi'
    model_path = dir_path / 'model'
    if read_name != '' and not read_name.endswith('_') :
        read_name = f'{read_name}_'
    module, dataset = read_mdb(model_path / f'{read_name}mdb.pkl')
    module.set_weights(read_sol_file(gurobi_path / f'{read_name}.sol'))
    return module, dataset

def write_files(module : LinealNN, dataset : CrossvalidationDataset,
                dir_path : Path, save_name : str = '',
                zero_tolerance : float = 0.00001, min_bits : int = 8,
                exists_ok = True
) :
    """
    Temp
    """
    gurobi_path = dir_path / 'gurobi'
    model_path = dir_path / 'model'
    gurobi_path.mkdir(parents = True, exist_ok = True)
    model_path.mkdir(parents = True, exist_ok = True)
    if save_name != '' and not save_name.endswith('_') :
        save_name = f'{save_name}_'
    mdb_path = model_path / f'{save_name}mdb.pkl'
    files_suffix = ['bias', 'exp', 'bits', 'mask', 'arch', 'lReLU', 'cls_tgt', 'reg_tgt', 'ftr']
    files_path_dict = {
        file_type : gurobi_path / f'{save_name}{file_type}.csv'
        for file_type in files_suffix
    }
    assert exists_ok or all(
        not file_path.exists()
        for file_path in [mdb_path, *files_path_dict.values]
    ), 'Algunos de los archivos ya existe'
    write_mdb(mdb_path, module, dataset)
    layers, act_layers = module.layers, module.activation_layers
    act_func = module.hyperparams['activation function']
    DataFrame(dataset.class_targets).to_csv(
        files_path_dict['cls_tgt'], header = False, index = False
    )
    DataFrame(dataset.regression_targets).to_csv(
        files_path_dict['reg_tgt'], header = False, index = False
    )
    DataFrame(dataset.features).to_csv(
        files_path_dict['ftr'], header = False, index = False
    )
    weights = module.get_weights()
    mantissa, exponent  = zip(*( numpy.frexp(weight.T) for weight in weights ))
    exponent = list(exponent)
    bits = [
        numpy.minimum(
            numpy.floor_divide(1, m.log2()),
            numpy.full_like(m, min_bits)
        ) for m in mantissa
    ]
    connections = [ (numpy.absolute(weight).T > zero_tolerance) for weight in weights ]
    leaky_relu = [ act_layers[k].weight.cpu().detach().numpy()
        if k in act_layers and isinstance(act_layers[k], PReLU)
        else numpy.full((module.capacity[k + 1],), act_layers[k].negative_slope)
        if k in act_layers and isinstance(act_layers[k], LeakyReLU)
        else numpy.full((module.capacity[k + 1],), 0.25)
        for k in (str(k) for k in range(layers + 1))
    ]
    tensor_list_to_dataframe(leaky_relu).to_csv(
        files_path_dict['lReLU'], header = True, index = False
    )
    tensor_list_to_dataframe(connections).to_csv(
        files_path_dict['mask'], header = True, index = False
    )
    tensor_list_to_dataframe(bits).to_csv(
        files_path_dict['bits'], header = True, index = False
    )
    tensor_list_to_dataframe(exponent).to_csv(
        files_path_dict['exp'], header = True, index = False
    )
    arch = {}
    arch['cap'] = module.capacity
    arch['act'] = [
        act_func[k] if k in act_func else None
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
        else None
        for k in (
            str(k) for k in range(layers + 1)
        )
    ]
    arch['ht_max'] = [
        act_layers[k].max_val
        if k in act_layers and isinstance(act_layers[k], Hardtanh)
        else None
        for k in (
            str(k) for k in range(layers + 1)
        )
    ]
    l1w = module.hyperparams['l1w']
    arch['l1w'] = [
        l1w[k] if k in l1w else None
        for k in range(layers + 1)
    ]
    l1a = module.hyperparams['l1a']
    arch['l1a'] = [
        l1a[k] if k in l1a else None
        for k in range(layers + 1)
    ]
    bias = module.hyperparams['bias init']
    arch['bias'] = [
        bias[k] if k in bias else None
        for k in range(layers + 1)
    ]
    DataFrame(arch).to_csv(
        files_path_dict['arch'], header = True, index = False
    )

def main(args : argparse.Namespace) :
    """
    Funcion main para generar la base de datos en archivo csv del archivo
    json del kardex anonimizado
    """
    load_path = Path(args.load_path)
    save_path = Path(args.save_path)
    if not save_path.exists() :
        print(f'{save_path} doesn\'t exists')
        return None
    elif not save_path.is_dir() :
        print(f'Cannot access {save_path} or isn\'t a directory')
        return None
    if not load_path.exists() :
        print(f'{load_path} doesn\'t exists')
        return None
    elif not load_path.is_dir() :
        print(f'Cannot access {load_path} or isn\'t a directory')
        return None
    file_prefix = args.case_name
    if not args.from_gurobi :
        models_path = load_path / 'mdbs'
        gurobi_path = save_path / 'gurobi'
        if not models_path.exists() :
            print(f'{models_path} doesn\'t exists')
            return None
        elif not models_path.is_dir() :
            print(f'Cannot access {models_path} or isn\'t a directory')
            return None
        module, dataset = read_mdb(models_path / f'{safe_suffix(file_prefix,'mdb')}.pkl')
        write_files(
            module  = module, dataset = dataset,
            dir_path = save_path, save_name = file_prefix,
            zero_tolerance = args.zero_tolerance,
            min_bits = args.min_bits,
            exists_ok = not args.no_overwrite
        )
    else :
        models_path = save_path / 'mdbs'
        gurobi_path = load_path / 'gurobi'
        if not gurobi_path.exists() :
            print(f'{gurobi_path} doesn\'t exists')
            return None
        elif not gurobi_path.is_dir() :
            print(f'Cannot access {gurobi_path} or isn\'t a directory')
            return None
        if file_prefix != '' and not file_prefix.endswith('_') :
            file_prefix = f'{file_prefix}_'
        module, dataset = read_files(load_path, file_prefix)
        models_path.mkdir(parents = True, exist_ok = True)
        model_name = safe_suffix(file_prefix,'gb')
        mdb_path = models_path / f'{model_name}_mdb.pkl'
        assert not(
            args.no_overwrite and mdb_path.exists()
        ), f'El archivo {mdb_path} ya existe'
        save_module(module,models_path,model_name, not args.no_overwrite)
        write_mdb(mdb_path,module,dataset)

if __name__ == '__main__' :
    import sys
    argparser = ArgumentParser()
    argparser.add_argument('--from_gurobi', '-fg', action = 'store_true')
    argparser.add_argument('--save_path', '-sp', default = Path.cwd())
    argparser.add_argument('--load_path', '-lp', default = Path.cwd())
    argparser.add_argument('--zero_tolerance', '-zt', default = 0.000001)
    argparser.add_argument('--min_bits', '-mb', default = 8)
    argparser.add_argument('--no_overwrite', '-eo', action = 'store_true')
    argparser.add_argument('--case_name', '-cn', default = '')
    sys.exit(main(argparser.parse_args(sys.argv[1:])))
