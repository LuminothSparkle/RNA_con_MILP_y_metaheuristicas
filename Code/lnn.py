import torch
from torch.nn import (Sequential, Module, Linear, BatchNorm1d, Dropout, ReLU, ReLU6, PReLU,
                      LeakyReLU, Hardsigmoid, Hardtanh, Tanh, Sigmoid, LogSoftmax, Softmax, 
                      Softmin, Softplus, Mish, SiLU, ELU, GELU, CELU, RReLU, SELU, Hardswish,
                      Identity, LogSigmoid, Softsign, ModuleDict, ModuleList)
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler
from torch.masked import as_masked_tensor
from torch.nn.init import xavier_normal_, xavier_uniform_, kaiming_uniform_, uniform_, trunc_normal_, orthogonal_, sparse_, kaiming_normal_, normal_
from torch.nn.utils import fuse_linear_bn_weights
from torch.nn.functional import dropout

from collections.abc import Iterable, Callable
from typing import Any
from numpy import ndarray, array
import numpy

class LinealNN(Module):
    functional_dict : dict[str,type[Module]] = {
                    'ReLU' : ReLU, 
                    'LeakyReLU' : LeakyReLU,
                    'ReLU6' : ReLU6,
                    'Hardsigmoid' : Hardsigmoid,
                    'Hardtanh' : Hardtanh,
                    'PReLU' : PReLU,
                    'Sigmoid' : Sigmoid,
                    'Softmin' : Softmin,
                    'Softmax' : Softmax,
                    'LogSoftmax' : LogSoftmax,
                    'Tanh' : Tanh,
                    'Softsign' : Softsign,
                    'Softplus' : Softplus,
                    'Mish' : Mish,
                    'SiLU' : SiLU,
                    'GELU' : GELU,
                    'CELU' : CELU,
                    'RReLU' : RReLU,
                    'SELU' : SELU,
                    'LogSigmoid' : LogSigmoid,
                    'Hardswish' : Hardswish,
                    'ELU' : ELU,
                    'None' : Identity
    }
    init_dict : dict[str,Callable[[Tensor,],Tensor]] = {
        'xavier uniform' : xavier_uniform_,
        'xavier normal' : xavier_normal_,
        'kaiming uniform' : kaiming_uniform_,
        'kaiming normal' : kaiming_normal_,
        'uniform' : uniform_,
        'normal' : normal_,
        'trunc normal' : trunc_normal_,
        'sparse' : sparse_, # type: ignore
        'orthogonal' : orthogonal_
    }
    dropout_layers : ModuleDict
    batch_norm_layers : ModuleDict
    linear_layers : ModuleDict
    activation_layers : ModuleDict
    sequential_layers : ModuleDict
    C : list[int]; L : int; phi : dict[int,str]
    act : list[Tensor]
    best_model_dict : dict[str,Any]
    weights_mask : dict[int,Tensor]
    bias_mask : dict[int,Tensor]
    hyperparams : dict[str,Any]
        
    def __init__(self, C : Iterable[int], hyperparams : dict[str,Any] = {}) :
        super().__init__()
        self.C = list(C); self.L = len(self.C) - 1
        self.hyperparams = { 'C' : list(C) }
        self.hyperparams['L'] =  len(self.hyperparams['C']) - 1
        for param in ['bias init', 'weight init', 'dropout', 'batch normalization', 'activation function', 'weight mask','connect dropout', 
                      'l1 weight regularization', 'l2 weight regularization', 'l1 activation regularization', 'l2 activation regularization'] :
            self.hyperparams[param] = {}
            if param in hyperparams and hyperparams[param] is not None :
                if isinstance(hyperparams[param], dict) :
                    self.hyperparams[param] = hyperparams[param]
                elif isinstance(hyperparams[param], list) :
                    self.hyperparams[param] = {k : data for k,data in enumerate(hyperparams[param])}
                else :
                    self.hyperparams[param] = {k : hyperparams[param] for k in range(self.L)}
        with torch.device('cuda') :
            self.bias_init = [ self.hyperparams['bias init'][k] if k in self.hyperparams['bias init'] else torch.ones for k in range(self.L) ]
            self.sequential_layers = ModuleDict({ str(k) : Sequential() for k in range(self.L) } )
            self.dropout_layers = ModuleDict({ str(k) : Dropout(p, inplace = True) for k,p in self.hyperparams['dropout'].items() })
            self.batch_norm_layers = ModuleDict({ str(k) : BatchNorm1d(num_features = self.C[k] + 1,**dict(kw)) for k,kw in self.hyperparams['batch normalization'].items() })
            self.linear_layers = ModuleDict({ str(k) : Linear(self.C[k] + 1,self.C[k + 1], bias = False) for k in range(self.L) })
            self.activation_layers = ModuleDict({ str(k) : self.functional_dict[name](**kw) for k,(name,kw) in self.hyperparams['activation function'].items() }) # type: ignore
            for weight,init in ((self.linear_layers[k].weight, init) for k,init in self.hyperparams['weight init'].items()) :
                if isinstance(init,tuple) :
                    init,kw = init
                    self.init_dict[init](weight.data, **kw) # type: ignore
                elif isinstance(init, Tensor) :
                    weight = init
            for weight,mask in ((self.linear_layers[k].weight, mask) for k,mask in self.hyperparams['weight mask'].items()) :
                weight = as_masked_tensor(data = weight, mask = mask)
            for k,drop in self.dropout_layers.items() :
                self.sequential_layers[k].add_module(f'Dropout {k}',drop)
            for k,batch in self.batch_norm_layers.items() :
                self.sequential_layers[k].add_module(f'Batch normalization {k}',batch)
            for (k,seq),linear in zip(self.sequential_layers.items(),self.linear_layers.values()) :
                seq.add_module(f'Linear {k}',linear)
            for k,act in self.activation_layers.items() :
                self.sequential_layers[k].add_module(f'Activation {k}',act)
            for k,layer in self.sequential_layers.items() :
                self.add_module(f'Sequential {k}', layer)

    def forward(self, input_tensor : Tensor) -> Tensor :
        with torch.device('cuda') :
            X = input_tensor.cuda()
            bias = torch.ones((X.size(dim = 0), self.hyperparams['L']))
            if self.training :
                for k,b_init in self.hyperparams['bias init'].items() :
                    bias[:,k] = b_init(X.size(dim = 0))
            self.act = []
            for k,layer in enumerate(self.sequential_layers.values()) :
                X = layer(torch.concat((X,bias[:,k].unsqueeze(dim = 1)), dim = 1))
                self.act += [X]
        return X

    def train_loop(self, dataloader : DataLoader, loss_fn : Callable[[Tensor,Tensor],Tensor], optimizer : Optimizer, epochs : int, scheduler : LRScheduler | None = None, 
                   verbose : bool = False, test_dataloader : DataLoader | None = None, early_tolerance : int | None = None):
        loss = { }
        loss['train'] = ndarray((epochs, len(dataloader)), dtype = float)
        loss['normalized'] = ndarray((epochs, len(dataloader)), dtype = float)
        loss['mean'] = ndarray((epochs), dtype = float)
        loss['best'] = float('inf')
        if test_dataloader is not None :
            loss['test'] = ndarray((epochs, len(test_dataloader)), dtype = float)
            with torch.inference_mode() :
                self.eval()
                loss['start test'] = array( [loss_fn(self(X), y.cuda()).item() for X,y in test_dataloader] ) # type: ignore
                self.train()
            if verbose :
                print(f"start     avg test  loss : {numpy.mean(loss['start test']):>10g}") # type: ignore
        else :
            loss['test'] = None
        early_counter = 0
        self.train()
        for epoch in range(epochs) :
            for batch, (X, y) in enumerate(dataloader) :
                optimizer.zero_grad()
                w_original = {}
                for k,weight,p in ((k,self.linear_layers[k].weight,p) for k,p in self.hyperparams['connect dropout'].items()) :
                    w_original[k] = weight
                    weight = (1 - p) * dropout( input = weight, p = p, training = True, inplace = False ) # type: ignore
                batch_loss = loss_fn(self(X), y.cuda())
                loss['train'][epoch,batch] = batch_loss.item()
                for k,w_ori in w_original.items() :
                    self.linear_layers[k].weight = w_ori
                for k,l1w in self.hyperparams['l1 weight regularization'].items() :
                    for param in self.sequential_layers[k].parameters() :
                        batch_loss += param.abs().mean() * l1w
                for k,l2w in self.hyperparams['l2 weight regularization'].items() :
                    for param in self.sequential_layers[k].parameters() :
                        batch_loss += param.square().mean() * l2w / 2    
                for k,l1a in self.hyperparams['l1 activation regularization'].items() :
                    if k < self.hyperparams['L'] :
                        batch_loss += self.act[k].abs().mean() * l1a    
                for k,l2a in self.hyperparams['l2 activation regularization'].items() :
                    if k < self.hyperparams['L'] :
                        batch_loss += self.act[k].square().mean() * l2a / 2
                loss['normalized'][epoch,batch] = batch_loss.item()
                batch_loss.backward()
                optimizer.step()
            if verbose :
                print(f"epoch {epoch:>3d} avg train loss : {loss['train'][epoch,:].mean():>10g}")
                print(f"epoch {epoch:>3d} avg norm  loss : {loss['normalized'][epoch,:].mean():>10g}")
            if scheduler is not None :
                scheduler.step()
            if test_dataloader is not None :
                with torch.inference_mode() :
                    self.eval()
                    loss['test'][epoch,:] = array( [loss_fn(self(X), y.cuda()).item() for X,y in test_dataloader] ) # type: ignore
                    loss['mean'][epoch] = loss['test'][epoch,:].mean() # type: ignore
                    self.train()
                if verbose :
                    print(f"epoch {epoch:>3d} avg test  loss : {loss['mean'][epoch]:>10g}") # type: ignore
            else :
                loss['mean'][epoch] = loss['train'][epoch,:].mean()
            if loss['mean'][epoch] < loss['best'] :
                early_counter = 0
                loss['best'] = loss['mean'][epoch]
                loss['model dict'] = self.state_dict().copy()
            elif early_tolerance is not None : 
                if early_counter < early_tolerance :
                    early_counter += 1
                else :
                    loss['train'] = loss['train'][:(epoch+1),:]
                    if loss['test'] is not None :
                        loss['test'] = loss['test'][:(epoch+1),:] # type: ignore
                    break
        self.eval()
        return loss
    
    def test_loop(self, dataloader : DataLoader, loss_fn : Callable[[Tensor,Tensor],Tensor], verbose : bool = False) :
        with torch.device('cuda') :
            with torch.inference_mode() :
                self.eval()
                test_loss = array(loss_fn(self(X), y).item() for X,y in dataloader)
        if verbose :
            print(f"avg test loss: {test_loss.mean():>8f} \n")
        return test_loss

    def get_weights(self) :
        with torch.inference_mode() :
            self.eval()
            weights = []
            for k,linear in self.linear_layers.items() :
                w, b = torch.tensor_split( linear.weight, (linear.weight.size(dim = 1) - 1,), dim = 1 ) # type: ignore
                if k in self.batch_norm_layers :
                    bn = self.batch_norm_layers[k]
                    w, b = fuse_linear_bn_weights(w, b, bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias) # type: ignore
                weights += [ torch.stack((w,b.unsqueeze(dim = 1)), dim = 1).cpu().numpy() ]
        return weights