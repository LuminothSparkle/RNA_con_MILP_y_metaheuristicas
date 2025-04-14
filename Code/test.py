import torch
import os.path as path
import sys
from pandas import read_csv
from torch.utils.data import TensorDataset
import torch.utils.data as data
from torch import Tensor
import nnlib as nnl
import utility.nn.torchlib as tlib
import SA_torch as SA
from pathlib import Path
from time import perf_counter_ns

if __name__ == '__main__' :
    data_path = path.normpath(sys.argv[1])
    tensor_path = path.join(data_path,'features.csv')
    label_path = path.join(data_path,'target.csv')
    tensor_dataframe = read_csv(tensor_path, header = 0, index_col = 0)
    label_dataframe = read_csv(label_path, header = 0, index_col = 0)
    torch.set_default_dtype(torch.double)
    batch_size = 64
    dataset = TensorDataset( torch.tensor( tensor_dataframe.values, dtype = torch.double ), torch.tensor( label_dataframe.values, dtype = torch.double ) )
    #epochs = 1000
    #iterations = 1
    epochs = 10000000
    iterations = 100
    lr = 0.00001
    optimizer = lambda parameters : torch.optim.SGD(parameters, lr = lr)
    loss_fn = torch.nn.SmoothL1Loss()
    C_0 = tensor_dataframe.shape[1]
    C_L = label_dataframe.shape[1]
    C = [C_0, 2 * C_0, C_0, C_L]
    phi = ['PReLU','PReLU','PReLU']
    arch = nnl.NN_Architecture(C,phi)
    prelu_ops = [{'num_parameters' : C[1]}, {'num_parameters' : C[2]}, {'num_parameters' : C[3]}]
    arch_eva = nnl.ArchEvaluator(train_dataset=dataset,optimizer=optimizer,loss_fn=loss_fn,epochs=epochs,batch_size=batch_size,iterations=iterations,act_ops = prelu_ops)
    ns_i = perf_counter_ns()
    loss = arch_eva(arch)
    ns_t = perf_counter_ns()
    if arch_eva.best_module is not None : 
        w,b = arch_eva.best_module.get_weights()
        digits = [4 * torch.ones(C[k] + 1,C[k + 1]) for k in range(len(C) - 1)]
        prelu = arch_eva.best_module.get_PreLU_weights()
        gb_w = tlib.weights_bias2tensor(w,b)
        precision = tlib.weights_precision(gb_w,digits)
        weights_path = Path(sys.argv[1]) / 'weights.csv'
        bias_path = Path(sys.argv[1]) / 'bias.csv'
        gurobi_path = Path(sys.argv[1]) / 'gb_w.csv'
        train_path = Path(sys.argv[1]) / 'train.log'
        arch_path = Path(sys.argv[1]) / 'arch.csv'
        prelu_path = Path(sys.argv[1]) / 'PReLU.csv'
        digits_path = Path(sys.argv[1]) / 'digits.csv'
        precision_path = Path(sys.argv[1]) / 'precision.csv'
        tlib.save_LMDB(w,weights_path, dtype = 'Float64')
        tlib.save_LVDB(b,bias_path, dtype = 'Float64')
        tlib.save_LMDB(gb_w,gurobi_path, dtype = 'Float64')
        tlib.save_HA(arch.C,arch.phi,arch_path)
        tlib.save_LMDB(digits, digits_path, dtype = 'Int64')
        tlib.save_LMDB(precision, precision_path, dtype = 'Int64')
        tlib.save_LVDB(prelu, prelu_path, dtype = 'Float64')
        with train_path.open(mode = 'wt', encoding = 'utf-8') as text_file :
            ns = ns_t - ns_i;
            print(f'loss : {loss}', file = text_file)
            print(f'best loss : {arch_eva.best_loss}', file = text_file)
            print(f'time : {ns} ns', file = text_file)
            ns, mus = ns % 1000, ns // 1000
            mus,ms = mus % 1000, mus // 1000
            ms, s = ms % 1000, ms // 1000
            s, m = s % 60, s // 60
            print(f'time : {m} m {s} s {ms} ms {mus} micros {ns} ns', file = text_file)

    #x_0 = NNF.NN_Architecture(hidden_C,C_0,C_L,phi)
    #y_0 = arch_opt(x_0)
    #init_temperature = 100
    #iterations = 100
    #def neighbor_fn(NeN : NNF.NN_Architecture) :
    #    return NeN.random_neighbor()
    #L_best, loss_best = SA.simulated_annealing(arch_opt, x_0, y_0, SA.defaultTemperature, init_temperature, neighbor_fn, SA.defaultProbability, iterations, 'MIN')
    #print(L_best, loss_best)
