from pandas import DataFrame
from pandas import read_csv
from pandas import NA
from pandas import Series
import pandas
from os import path
from torch import Tensor
from torch import flatten
import torch


def save_HA(hidden_C : list[int], phi : list[str], save_path : str) :
    hidden_C_series = Series(data = hidden_C, dtype = 'Int64', name = 'Neurons')
    phi_series = Series(data = phi, dtype = 'string', name = 'Activation Function')
    arch_dataframe = pandas.merge(phi_series,hidden_C_series, how = 'outer', left_index = True, right_index = True)
    arch_dataframe.to_csv(path.normpath(save_path), index_label = 'Layer')

def load_HA(load_path : str) -> DataFrame :
    return read_csv(path.normpath(load_path), header = 0, index_col = 0, dtype = {'Neurons' : 'UInt64', 'Activation Function' : 'string'})

def save_TDB(tensor : Tensor, save_path : str, names : list[str] | None = None) :
    tensor_dataframe = DataFrame(data = tensor.tolist(), dtype = 'Float64', columns = names)
    tensor_dataframe.to_csv(path.normpath(save_path), index_label = 'Case')
    
def load_TDB(load_path : str) -> DataFrame :
    return read_csv(path.normpath(load_path), header = 0, index_col = 0, dtype = Tensor)
    
def save_FDB(weights : list[Tensor], save_path : str, index_label : str | None = None) :
    weights_frame = DataFrame()
    for weights_tensor in weights :
        dim_frame = DataFrame([weights_tensor.size()], columns = ['n', 'm'], dtype = 'Int64')
        matrix_frame = DataFrame([flatten(weights_tensor).tolist()], dtype = 'Float64')
        weights_row = pandas.concat([dim_frame,matrix_frame], axis = 'columns')
        weights_frame = pandas.concat([weights_frame,weights_row], axis = 'index', ignore_index = True, join = 'outer')
    weights_frame.to_csv(path.normpath(save_path), index_label = index_label)
    
def save_IDB(weights : list[Tensor], save_path : str, index_label : str | None = None) :
    weights_frame = DataFrame()
    for weights_tensor in weights :
        dim_frame = DataFrame([weights_tensor.size()], columns = ['n', 'm'], dtype = 'Int64')
        matrix_frame = DataFrame([flatten(weights_tensor).tolist()], dtype = 'Int64')
        weights_row = pandas.concat([dim_frame,matrix_frame], axis = 'columns')
        weights_frame = pandas.concat([weights_frame,weights_row], axis = 'index', ignore_index = True, join = 'outer')
    weights_frame.to_csv(path.normpath(save_path), index_label = index_label)
    
def load_FDB(load_path : str) :
    weights_dataframe = read_csv(path.normpath(load_path))
    L, _ = weights_dataframe.shape
    weights = []
    for k in range(L) :
        n, m = weights_dataframe.loc[k,['n','m']].astype('Int64')
        weights += [ torch.tensor( [ *weights_dataframe.loc[k,[str(i) for i in range(n*m)]].astype('Float64') ] ).view([n,m]) ]
    return weights

def load_IDB(load_path : str) :
    weights_dataframe = read_csv(path.normpath(load_path))
    L, _ = weights_dataframe.shape
    weights = []
    for k in range(L) :
        n, m = weights_dataframe.loc[k,['n','m']].astype('Int64')
        weights += [ torch.tensor( [ *weights_dataframe.loc[k,[str(i) for i in range(n*m)]].astype('Int64') ] ).view([n,m]) ]
    return weights

def load_sol(load_path : str) -> DataFrame :
    return read_csv(path.normpath(load_path), skiprows = 1, sep = ' ', header = None, index_col = 0, names = ['Variable', 'Value'])


#def load_weights_from_sol(load_path : str) -> list[Tensor] :
#    sol_frame = load_sol(load_path)
#    binaries_series = Series(dtype = 'Int64')
#    layer_frame = DataFrame(columns = ['n', 'm'], dtype = 'Int64').fillna(0)
#    for index, value in sol_frame.itertuples(index =  True, name = None) :
#        if index.startswith('b_') :
#            k,i,j,l = [ int(string) for string in index.strip().lstrip('b_').split('_') ]
#            n,m = layer_frame.get(k,[0,0])
#            layer_frame[k][['n','m']] = [max(i,n), max(j,m)]
#            binaries_series[(k,i,j,l)] = int(value)
#    for k,i,j,l,value in binaries_series.itertuples(index =  True, name = None) :
        
    
    