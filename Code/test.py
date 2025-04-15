import torch
import os.path as path
import sys
from pandas import read_csv
from torch.utils.data import TensorDataset
import nnlib as nnl
import utility.nn.torchlib as tlib
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
    X, Y = torch.tensor( tensor_dataframe.values, dtype = torch.double, device = 'cuda' ), torch.tensor( label_dataframe.values, dtype = torch.double, device = 'cuda' )
    dataset = TensorDataset( X,Y )
    #epochs = 1000
    #iterations = 1
    epochs = 10000
    iterations = 1
    lr = 0.00001
    optimizer = lambda parameters : torch.optim.SGD(parameters, lr = lr)
    loss_fn = torch.nn.SmoothL1Loss()
    C_0 = tensor_dataframe.shape[1]
    C_L = label_dataframe.shape[1]
    C = [C_0, 2 * C_0, C_0, C_0 // 2, C_L]
    phi = ['PReLU','PReLU','PReLU','PReLU']
    prelu_ops = [{'num_parameters' : C[1]}, {'num_parameters' : C[2]}, {'num_parameters' : C[3]}, {'num_parameters' : C[4]}]
    ns_i = perf_counter_ns()
    results, best_train, best_test, best_all = nnl.shuffle_iterate(dataset,optimizer,loss_fn,C,phi,epochs,iterations,train_batches = 10,test_percentage = 0.33, act_kwargs = prelu_ops)
    ns_t = perf_counter_ns()
    ns = ns_t - ns_i
    ns, mus = ns % 1000, ns // 1000
    mus,ms = mus % 1000, mus // 1000
    ms, s = ms % 1000, ms // 1000
    s, m = s % 60, s // 60
    train_path = Path(sys.argv[1]) / 'train.log'
    with train_path.open(mode = 'wt', encoding = 'utf-8') as text_file :
        print(f'time : {ns} ns', file = text_file)
        print(f'time : {m} m {s} s {ms} ms {mus} micros {ns} ns', file = text_file)
        if best_test is not None :
            module = results[best_test[0]][2]
            with torch.no_grad() :
                pred = module(X).cpu().detach()
                comp = torch.cat((Y.cpu(), pred), dim = 1)
                w, b = module.get_weights_bias()
                digits = [4 * torch.ones(C[k] + 1,C[k + 1]).detach() for k in range(len(C) - 1)]
                prelu = module.get_PreLU_weights()
                gb_w = tlib.weights_bias2tensor(w,b)
                precision = tlib.weights_precision(gb_w,digits)
                weights_path = Path(sys.argv[1]) / 'best_test_weights.csv'
                bias_path = Path(sys.argv[1]) / 'best_test_bias.csv'
                gurobi_path = Path(sys.argv[1]) / 'gb_w.csv'
                arch_path = Path(sys.argv[1]) / 'best_test_arch.csv'
                prelu_path = Path(sys.argv[1]) / 'best_test_PReLU.csv'
                digits_path = Path(sys.argv[1]) / 'best_test_digits.csv'
                precision_path = Path(sys.argv[1]) / 'best_test_precision.csv'
                prediction_path = Path(sys.argv[1]) / 'best_test_prediction.csv'
                tlib.save_LMDB(w,weights_path, dtype = 'Float64')
                tlib.save_LVDB(b,bias_path, dtype = 'Float64')
                tlib.save_LMDB(gb_w,gurobi_path, dtype = 'Float64')
                tlib.save_HA(C,phi,arch_path)
                tlib.save_LMDB(digits, digits_path, dtype = 'Int64') # type: ignore
                tlib.save_LMDB(precision, precision_path, dtype = 'Int64')
                tlib.save_LVDB(prelu, prelu_path, dtype = 'Float64') # type: ignore
                tlib.save_TDB(pred,prediction_path)
                print(f'best test loss : {best_test[1]}', file = text_file)
        if best_train is not None :
            module = results[best_train[0]][2]
            with torch.no_grad() :
                pred = module(X).cpu().detach()
                comp = torch.cat((Y.cpu(), pred), dim = 1)
                w, b = module.get_weights_bias()
                digits = [4 * torch.ones(C[k] + 1,C[k + 1]).detach() for k in range(len(C) - 1)]
                prelu = module.get_PreLU_weights()
                gb_w = tlib.weights_bias2tensor(w,b)
                precision = tlib.weights_precision(gb_w,digits)
                weights_path = Path(sys.argv[1]) / 'best_train_weights.csv'
                bias_path = Path(sys.argv[1]) / 'best_train_bias.csv'
                prelu_path = Path(sys.argv[1]) / 'best_train_PReLU.csv'
                prediction_path = Path(sys.argv[1]) / 'best_train_prediction.csv'
                tlib.save_LMDB(w,weights_path, dtype = 'Float64')
                tlib.save_LVDB(b,bias_path, dtype = 'Float64')
                tlib.save_LVDB(prelu, prelu_path, dtype = 'Float64') # type: ignore
                tlib.save_TDB(pred,prediction_path)
                print(f'best test loss : {best_train[1]}', file = text_file)
        if best_all is not None :
            module = results[best_all[0]][2]
            with torch.no_grad() :
                pred = module(X).cpu().detach()
                comp = torch.cat((Y.cpu(), pred), dim = 1)
                w, b = module.get_weights_bias()
                digits = [4 * torch.ones(C[k] + 1,C[k + 1]).detach() for k in range(len(C) - 1)]
                prelu = module.get_PreLU_weights()
                gb_w = tlib.weights_bias2tensor(w,b)
                precision = tlib.weights_precision(gb_w,digits)
                weights_path = Path(sys.argv[1]) / 'best_all_weights.csv'
                bias_path = Path(sys.argv[1]) / 'best_all_bias.csv'
                prelu_path = Path(sys.argv[1]) / 'best_all_PReLU.csv'
                prediction_path = Path(sys.argv[1]) / 'best_all_prediction.csv'
                tlib.save_LMDB(w,weights_path, dtype = 'Float64')
                tlib.save_LVDB(b,bias_path, dtype = 'Float64')
                tlib.save_LVDB(prelu, prelu_path, dtype = 'Float64') # type: ignore
                tlib.save_TDB(pred,prediction_path)
                print(f'best test loss : {best_all[1]}', file = text_file)

    #x_0 = NNF.NN_Architecture(hidden_C,C_0,C_L,phi)
    #y_0 = arch_opt(x_0)
    #init_temperature = 100
    #iterations = 100
    #def neighbor_fn(NeN : NNF.NN_Architecture) :
    #    return NeN.random_neighbor()
    #L_best, loss_best = SA.simulated_annealing(arch_opt, x_0, y_0, SA.defaultTemperature, init_temperature, neighbor_fn, SA.defaultProbability, iterations, 'MIN')
    #print(L_best, loss_best)
