import torch
import pandas
from pandas import DataFrame, Index, read_csv
from pathlib import Path
from Code.lnn import LinealNN
from torch import Tensor
from collections.abc import Sequence
from torch.nn import Hardtanh, PReLU, Dropout

def tensor_list_to_dataframe(tensor_list : list[Tensor | None]) :
    dim_rows = []
    tensor_rows = []
    for tensor in tensor_list :
        if tensor is not None :
            dim_rows += [DataFrame(data = [tensor.size()], columns = [f'd_{dim}' for dim in tensor.shape ], dtype = 'Int64')]
            tensor_rows += [DataFrame(data = [tensor.t().flatten().tolist()], dtype = tensor.dtype)]
        else :
            dim_rows += [DataFrame()]
            tensor_rows += [DataFrame()]
    dim_dataframe = pandas.concat(dim_rows, axis = 'index', ignore_index = True, join = 'outer').fillna(0)
    tensor_dataframe = pandas.concat(tensor_rows, axis = 'index', ignore_index = True, join = 'outer')
    return pandas.concat([dim_dataframe,tensor_dataframe], axis = 'columns', ignore_index = False, join = 'outer')

def dataframe_to_tensor_list(dataframe : DataFrame) :
    dim = len(dataframe.columns.str.startswith('d_'))
    dim_index = Index([ f'd_{d}' for d in range(dim) ])
    data_index = dataframe.columns.drop(dim_index)
    tensor_list = []
    for dim,data in zip(dataframe[dim_index].to_numpy(), dataframe[data_index].to_numpy()) :
        tensor_list += [torch.tensor(data).view(dim).t()]
    return tensor_list

def arch_to_dataframe(C : Sequence[int], phi : Sequence[str| None], l1_weights_norm : Sequence[float | None] | float | None = None, l1_activation_norm : Sequence[float | None] | float | None = None, hardtanh_limits : Sequence[tuple[float,float] | tuple[None,None]] | tuple[float,float] | None = None) :
    assert len(C) - 1 == len(phi), ''
    assert isinstance(l1_activation_norm,Sequence) and len(phi) == len(l1_activation_norm), ''
    assert isinstance(l1_weights_norm,Sequence) and len(phi) == len(l1_weights_norm), ''
    assert isinstance(hardtanh_limits,Sequence) and len(phi) == len(hardtanh_limits), ''
    if hardtanh_limits is None :
        hardtanh_data = { 'ht_min' : [],  'ht_min' : [] }
    elif isinstance(hardtanh_limits,tuple) :
        hardtanh_data = {'ht_min' : [hardtanh_limits[0] for _ in range(len(phi))], 'ht_min' : [hardtanh_limits[1] for _ in range(len(phi))]}
    else :
        hardtanh_data = { 'ht_min' : [htl[0] if htl is not None else htl for htl in hardtanh_limits] , 'ht_max' : [htl[1] if htl is not None else htl for htl in hardtanh_limits] }
    if l1_activation_norm is None :
        l1a_data = { 'l1a' : [] }
    elif isinstance(l1_activation_norm,float) :
        l1a_data = { 'l1a' : [l1_activation_norm for _ in range(len(phi))] }
    else :
        l1a_data = { 'l1a' : l1_activation_norm }
    if l1_weights_norm is None :
        l1w_data = { 'l1w' : [] }
    elif isinstance(l1_weights_norm,float) :
        l1w_data = { 'l1w' : [l1_weights_norm for _ in range(len(phi))] }
    else :
        l1w_data = { 'l1w' : l1_weights_norm }
    ht_dataframe = DataFrame(hardtanh_data)
    l1a_dataframe = DataFrame(l1a_data)
    l1w_dataframe = DataFrame(l1w_data)
    C_dataframe = DataFrame({'C' : C})
    phi_dataframe = DataFrame({'phi' : phi})    
    return pandas.concat([C_dataframe,phi_dataframe,l1w_dataframe,l1a_dataframe,ht_dataframe], axis = 'columns', join = 'outer')

def read_sol_file(file_path : Path) :
    sol_dataframe = read_csv(filepath_or_buffer = file_path, sep = ' ', index_col = 0, skiprows = 1, header = None)
    w_var_dataframe = sol_dataframe.loc[sol_dataframe.index.str.startswith('w_'),:]
    L = 1
    for name in w_var_dataframe.index :
        _,k,_,_ = name.split('_')
        L = max(int(k) + 1,L)
    dim_list = [[1,1] for _ in range(L)]
    for name in w_var_dataframe.index :
        _,k,i,j = name.split('_')
        dim_list[int(k)][0] = max(dim_list[int(k)][0], int(i) + 1)
        dim_list[int(k)][1] = max(dim_list[int(k)][1], int(j) + 1)
    w_list = [ torch.empty(dim) for dim in dim_list]
    for name, value in w_var_dataframe.itertuples(index = True) :
        _,k,i,j = name.split('_')
        w_list[int(k)][int(i),int(j)] = float(value)
    return w_list

class TorchGRBInterface :
    module : LinealNN
    class_targets : Tensor
    regression_targets : Tensor
    features : Tensor
    leakyReLU_slopes : list[Tensor | None]
    connections : list[Tensor | None]
    bits : list[Tensor | None]
    exponent : list[Tensor | None]
    C : list[int]
    phi : list[str | None]
    l1_weights_lambda : list[float | None]
    l1_activation_lambda : list[float | None]
    hardtanh_limits : list[tuple[float,float] | tuple[None,None]]
    dropout : list[float | None]
    
    def __init__(self, module : LinealNN) -> None:
        self.L = module.L
        self.leakyReLU_slopes = [ module.activation_layers[k].weight if k in module.activation_layers and isinstance(module.activation_layers[k], PReLU) else None for k in range(self.L + 1) ] # type: ignore
        self.dropout = [ module.activation_layers[k].p if k in module.activation_layers and isinstance(module.activation_layers[k], Dropout()) else None for k in range(self.L + 1) ] # type: ignore
        self.hardtanh_limits = [ (module.activation_layers[k].min_val, module.activation_layers[k].max_val) if k in module.activation_layers and isinstance(module.activation_layers[k], Hardtanh) else (None,None) for k in range(self.L + 1) ] # type: ignore
        self.phi = [module.phi[k] if k in module.phi else None for k in range(self.L + 1)]
        self.C = module.C
        self.l1_weights_lambda = [module.hyperparams['l1w'][k] if k in module.hyperparams['l1w'] else None for k in range(self.L + 1)]
        self.l1_activation_lambda = [module.hyperparams['l1a'][k] if k in module.hyperparams['l1a'] else None for k in range(self.L + 1)]
        weights = module.get_weights()
        mantissa, exponent  = zip(*( tensor.frexp() for tensor in weights))
        self.exponent = list(exponent)
        self.bits = [ torch.floor_divide(1,m.log2()).minimum(torch.full_like(m,8)) for m in mantissa ]
        self.connections = [ tensor.to(dtype = torch.bool) for tensor in weights ]

    def write_files(self, name : str = '', dir_path : Path | None = None, overwrite_error : bool = False) :
        if dir_path is None :
            dir_path = Path('')
        name = f'{name}_' if len(name) > 0 else '';
        files_suffix = ['exp', 'bits', 'mask', 'arch', 'lReLU', 'cls_tgt', 'reg_tgt', 'ftr']
        files_path_dict = { file_type : dir_path.joinpath(f'{name}{file_type}.csv') for file_type in files_suffix }
        if overwrite_error and any(path.exists() for path in files_path_dict.values()) :
            return
        DataFrame(self.class_targets.numpy()).to_csv(files_path_dict['cls_tgt'], header = False, index = False)
        DataFrame(self.regression_targets.numpy()).to_csv(files_path_dict['reg_tgt'], header = False, index = False)
        DataFrame(self.features.numpy()).to_csv(files_path_dict['ftr'], header = False, index = False)
        tensor_list_to_dataframe(self.leakyReLU_slopes).to_csv(files_path_dict['lReLU'], header = True, index = False)
        tensor_list_to_dataframe(self.connections).to_csv(files_path_dict['mask'], header = True, index = False)
        tensor_list_to_dataframe(self.bits).to_csv(files_path_dict['bits'], header = True, index = False)
        tensor_list_to_dataframe(self.exponent).to_csv(files_path_dict['exp'], header = True, index = False)
        arch_to_dataframe(self.C,self.phi,self.l1_weights_lambda,self.l1_activation_lambda,self.hardtanh_limits).to_csv(files_path_dict['arch'], header = True, index = False)