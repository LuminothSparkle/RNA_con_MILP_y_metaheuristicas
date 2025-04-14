import pandas
from pandas import (DataFrame, read_csv)
from pathlib import Path
import torch
from torch import (Tensor, flatten)

def save_HA(C : list[int], phi : list[str], save_path : Path) :
    assert len(C) - 1 == len(phi), 'Los tamaños de C y phi no concuerdan'
    columns = ['#Neu','Act Func']
    columns_types = ['Int64','string']
    arch_dataframe = DataFrame(index = range(len(C)), columns = columns)
    arch_dataframe = arch_dataframe.astype(dict(zip(columns,columns_types)), copy = False)
    arch_dataframe.iloc[range(len(C)),0] = C
    arch_dataframe.iloc[range(len(phi)),1] = phi
    arch_dataframe.to_csv(save_path, index_label = 'Layer')

def load_HA(load_path : Path) -> DataFrame :
    return read_csv(load_path, header = 0, index_col = 0, dtype = {'#Neu' : 'Int64', 'Act Func' : 'string'})

def save_TDB(tensor : Tensor, save_path : Path, columns : list[str] | None = None, index_label : str | None = None) :
    DataFrame(data = tensor.tolist(), columns = columns).to_csv(save_path, index_label = index_label)
    
def load_TDB(load_path : Path, columns : list[str] | None = None) -> DataFrame :
    return read_csv(load_path, header = 0, index_col = 0, names = columns)

def save_LVDB(vector_list : list[Tensor | None], save_path : Path, index_label : str | None = None, dtype = 'Float64') :
    rows = []
    for vector in vector_list :
        if vector is not None :
            dim_frame = DataFrame([vector.size()], columns = ['n'], dtype = 'Int64')
            vector_frame = DataFrame([vector.tolist()], dtype = dtype)
            rows += [ pandas.concat([dim_frame,vector_frame], axis = 'columns') ]
        else :
            rows += [ DataFrame([(0)], columns = ['n'], dtype = 'Int64') ]
    vector_frame = pandas.concat(rows, axis = 'index', ignore_index = True, join = 'outer')
    vector_frame.to_csv(save_path, index_label = index_label)

def load_LVDB(load_path : Path, dtype = torch.double) :
    vector_frame = read_csv(load_path)
    vector_list = []
    for k in vector_frame.index :
        n = vector_frame.at[k,'n']
        if n == 0 :
            vector_list += [ None ]
        else :
            vector_list += [ torch.zeros(n, dtype = dtype) ]
            vector_list[-1][:] = vector_frame.loc[k, [f'{i}' for i in range(n)] ]
    return vector_list
    
def save_LMDB(matrix_list : list[Tensor | None], save_path : Path, index_label : str | None = None, dtype = 'Float64') :
    rows = []
    for matrix in matrix_list :
        if matrix is not None :
            dim_frame = DataFrame([matrix.size()], columns = ['n', 'm'], dtype = 'Int64')
            matrix_frame = DataFrame([flatten(matrix).tolist()], dtype = dtype)
            rows += [ pandas.concat([dim_frame,matrix_frame], axis = 'columns') ]
        else :
            rows += [ DataFrame([(0,0)], columns = ['n', 'm'], dtype = 'Int64') ]
    matrix_frame = pandas.concat(rows, axis = 'index', ignore_index = True, join = 'outer')
    matrix_frame.to_csv(save_path, index_label = index_label)

def load_LMDB(load_path : Path, dtype = torch.double) :
    matrix_dataframe = read_csv(load_path)
    matrix_list = []
    for k in matrix_dataframe.index :
        n, m = matrix_dataframe.at[k,'n'], matrix_dataframe.at[k,'m']
        if n == 0 or m == 0 :
            matrix_list += [ None ]
        else :
            matrix_list += [ torch.zeros(n, m, dtype = dtype) ]
            matrix_list[-1][:] = matrix_dataframe.loc[k, [f'{i}' for i in range(n * m)] ]
    return matrix_list

def load_sol(load_path : Path) -> DataFrame :
    var_frame = read_csv(load_path, skiprows = 1, sep = ' ', header = None, index_col = 0, names = ['Variable', 'Value'], float_precision = 'round_trip')
    var_frame.set_index('Variable', inplace = True)
    return var_frame

def bin2weights_bias(var_frame : DataFrame, C : list[int], digits : list[Tensor], precision : list[Tensor] | None = None) :
    w = []
    for k in range(len(C) - 1) :
        w += [ torch.zeros(C[k] + 1, C[k + 1]) ]
        for i in range(C[k] + 1) :
            for j in range(C[k + 1]) :
                D = int(digits[k][i][j].item())
                if precision is not None :
                    w[k][i][j] = -pow(2, digits[k][i][j].item() - precision[k][i][j].item()) * var_frame.at[f'b_{k}_{i}_{j}_{D}','Value']
                    for l in range(D) :
                        w[k][i][j] += pow(2, l - precision[k][i][j].item()) * var_frame.at[f'b_{k}_{i}_{j}_{l}','Value']
                elif precision is None :
                    w[k][i][j] = -pow(2, digits[k][i][j].item()) * var_frame.at[f'b_{k}_{i}_{j}_{D}','Value']
                    for l in range(D) :
                        w[k][i][j] += pow(2, l) * var_frame.at[f'b_{k}_{i}_{j}_{l}','Value']
    return tensor2weights_bias(w)

def tensor2weights_bias(weights : list[Tensor | None]) :
    params = [[], []]
    for w in weights :
        if w is not None :
            weights, bias= w.tensor_split([-1], dim = 0)
            params[0] += [torch.t(weights)]; params[1] += [torch.t(bias)]
        else :
            params[0] += [None]; params[1] += [None]
    return tuple(params)

def weights_bias2tensor(weights : list[Tensor], bias : list[Tensor]) :
    weights_list = []
    for w,b in zip(weights,bias) :
        if w is not None and b is not None :
            weights_list += [ torch.t(torch.cat((w,b.unsqueeze(1)),1)) ]
        elif w is not None :
            weights_list += [ torch.t(w) ]
        elif b is not None :
            weights_list += [ torch.t(b) ]
        else :
            weights_list += [ None ]
    return weights_list

def weights_precision(weights : list[Tensor], digits : list[Tensor] | int) :
    precision = []
    if isinstance(digits,int) :
        for w in weights :
            _, e = w.frexp()
            precision += [ e - torch.div(digits + 1, 2, rounding_mode = 'floor').int() ]
    else :
        for w, d in zip(weights,digits) :
            _, e = w.frexp()
            precision += [ e - torch.div(d + 1, 2, rounding_mode = 'floor').int() ]
    return precision