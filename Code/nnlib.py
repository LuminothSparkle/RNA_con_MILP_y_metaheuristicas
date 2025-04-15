import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import BatchSampler
from torch.utils.data import (SequentialSampler, ConcatDataset)
import torch.utils.data as data
from random import randint
from collections.abc import (Sequence, Iterable, Sized, Mapping)
from typing import (Any)
import statistics
import random

functional_dict = {
                   'ReLU' : nn.ReLU, 
                   'LeakyReLU' : nn.LeakyReLU,
                   'ReLU6' : nn.ReLU6,
                   'Hardsigmoid' : nn.Hardsigmoid,
                   'Hardtanh' : nn.Hardtanh,
                   'PReLU' : nn.PReLU,
                   'None' : nn.Identity
}

def randomHiddenArch(L : int, max_C : int) :
    hidden_C = random.sample(range(1,max_C), L - 1)
    phi = random.choices([*functional_dict], k = L)
    return hidden_C, phi

def hiddenArch(L : int, C_n : int, phi_str : str) :
    hidden_C = [ C_n for _ in range(L - 1) ]
    phi = [ phi_str for _ in range(L) ]
    return hidden_C, phi

class LinealNN(nn.Module):
    def __init__(self,C : Sequence[int], phi : Sequence[str], weights : None | Sequence[tuple[Tensor | None, Tensor | None]] | Sequence[Tensor | None] = None, bias : None | Sequence[tuple[Tensor | None, Tensor | None]] | Sequence[Tensor | None] = None, act_kwargs : Sequence[dict[str,Any] | None] | None = None) :
        self.L = len(C)
        self.C = C
        self.phi = phi
        assert self.L - 1 == len(phi), 'Size of phi must be the size of C - 1'
        with torch.device('cuda') :
            super().__init__()
            self.train_loss, self.test_loss = [None] * 2
            self.model, self.linear_layers = nn.Sequential(), []
            self.activation_layers = []
            for k in range(self.L - 1) :
                self.linear_layers += [ nn.Linear(C[k], C[k + 1], True) ]
                if weights is not None :
                    matrix = weights[k]
                    if matrix is not None :
                        mask = None
                        if isinstance(matrix, tuple) :
                            matrix, mask = matrix
                        if matrix is not None :
                            self.linear_layers[-1].weight = matrix.cuda()
                        if mask is not None :
                            self.linear_layers[-1].weight *= mask.cuda()
                if bias is not None :
                    matrix = bias[k]
                    if matrix is not None :
                        mask = None
                        if isinstance(matrix, tuple) :
                            matrix, mask = matrix
                        if matrix is not None :
                            self.linear_layers[-1].weight = matrix.cuda()
                        if mask is not None :
                            self.linear_layers[-1].weight *= mask.cuda()
                if act_kwargs is not None and act_kwargs[k] is not None :
                    self.activation_layers += [ functional_dict[phi[k]](**act_kwargs[k]) ] # type: ignore
                else :
                    self.activation_layers += [ functional_dict[phi[k]]() ]
                self.model.append(self.linear_layers[-1]); self.model.append(self.activation_layers[-1])

    def forward(self, input_tensor : Tensor) -> Tensor:
        return self.model(input_tensor.cuda())

    def train_loop(self, dataloader : DataLoader, loss_fn : (...), optimizer : Optimizer, epochs : int, verbose : bool = False, weights_mask : None | Iterable[Tensor | None] = None, bias_mask : None | Iterable[Tensor | None] = None):
        w_cuda_mask : list[None | Tensor] = [None] * len(self.linear_layers)
        b_cuda_mask : list[None | Tensor] = [None] * len(self.linear_layers)
        if weights_mask is not None :
            for k, mask in enumerate(weights_mask) :
                if mask is not None :
                    w_cuda_mask[k] = mask.cuda()
        if bias_mask is not None :
            for k, mask in enumerate(bias_mask) :
                if mask is not None :
                    w_cuda_mask[k] = mask.cuda()
        # In cuda
        with torch.device('cuda') :
            self.model.train()
            self.train_loss = torch.zeros(epochs,len(dataloader)).detach()
            for epoch in range(epochs):
                for batch, (X, y) in enumerate(dataloader) :
                    pred = self.model(X.cuda(),)
                    loss = loss_fn(pred, y.cuda())
                    loss.backward()
                    for k, module in enumerate(self.linear_layers) :
                        if w_cuda_mask[k] is not None :
                            module.weight.grad *= w_cuda_mask[k]
                        if b_cuda_mask[k] is not None :
                            module.bias.grad *= b_cuda_mask[k]
                    optimizer.step(); optimizer.zero_grad()
                    self.train_loss[epoch,batch] = loss.item()
        # Out of cuda
        self.train_loss = self.train_loss.cpu()
        if verbose :
            for epoch in range(epochs):
                for batch in range(len(dataloader)) :
                    print(f"epoch {epoch} batch {batch} loss : {self.train_loss[epoch,batch].item():>7f}")
        return self.train_loss[-1,:].mean().item()

    def evaluate(self, dataloader : DataLoader, loss_fn : (...), verbose : bool = False) :
        # In cuda
        with torch.device('cuda') :
            self.model.eval()
            test_loss = torch.zeros(len(dataloader)).detach()
            with torch.no_grad() :
                for t, (X, y) in enumerate(dataloader) :
                    pred = self.model(X.cuda())
                    test_loss[t] = loss_fn(pred, y.cuda()).item()
        test_loss = test_loss.mean().item()
        if verbose :
            print(f"Avg test loss: {test_loss:>8f} \n")
        return test_loss

    def test_loop(self, dataloader : DataLoader, loss_fn : (...), verbose : bool = False) :
        # In cuda
        with torch.device('cuda') :
            self.model.eval()
            self.test_loss = torch.zeros(len(dataloader)).detach()
            with torch.no_grad() :
                for t, (X, y) in enumerate(dataloader) :
                    self.pred = self.model(X.cuda())
                    self.test_loss[t] = loss_fn(self.pred, y.cuda()).item()
        # Out of cuda
        self.test_loss = self.test_loss.cpu()
        if verbose :
            print(f"Avg test loss: {self.test_loss.mean():>8f} \n")
        return self.test_loss.mean().item()

    def get_weights_bias(self) :
        return zip(*( ( layer.weight.cpu().detach(), layer.bias.cpu().detach() ) for layer in self.linear_layers) )
    
    def get_PreLU_weights(self) :
        return [ layer.weight.cpu().detach() if isinstance(layer,nn.PReLU) else None for layer in self.activation_layers ]
    
def evaluate_arch(train_dataloader : DataLoader, optimizer : (...), loss_fn : (...), C : Sequence[int], phi : Sequence[str], epochs : int, test_dataloader : DataLoader | None, weights_mask : None | Iterable[Tensor | None] = None, bias_mask : None | Iterable[Tensor | None] = None, act_kwargs : None | Sequence[dict[str,Any] | None] = None, verbose : bool = False) :
    module = LinealNN(C = C,phi = phi, act_kwargs = act_kwargs)
    optimizer = optimizer(module.parameters())
    train_loss = module.train_loop(train_dataloader,loss_fn,optimizer,epochs, weights_mask = weights_mask, bias_mask = bias_mask, verbose = verbose)
    test_loss = None
    if test_dataloader is not None :
        test_loss  = module.test_loop(test_dataloader,loss_fn,verbose)
        if verbose :
            print(f"test loss: {test_loss:>8f}")
    return train_loss, test_loss, module, optimizer

def shuffle_iterate(dataset : Dataset, base_optimizer : (...), loss_fn : (...), C : Sequence[int], phi : Sequence[str], epochs : int, iterations : int, train_batches : int = 10, test_percentage : float = 0.3, weights_mask : None | Iterable[Tensor | None] = None, bias_mask : None | Iterable[Tensor | None] = None, act_kwargs : None | Sequence[dict[str,Any] | None] = None, verbose : bool = False) :
    results = []
    best_all = None
    best_train = None
    best_test = None
    dataset_size = len(dataset) # type: ignore
    dataloader = DataLoader(dataset, batch_size = dataset_size)
    test_size = int(dataset_size * test_percentage)
    train_size = dataset_size - test_size
    batch_size = train_size // train_batches
    for it in range(iterations) :
        train_dataset, test_dataset = data.random_split(dataset,[train_size, test_size])
        train_dataloader = DataLoader(train_dataset, sampler = torch.randperm(train_size ).tolist(), batch_size = batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size = test_size)
        results += [(evaluate_arch(train_dataloader,base_optimizer,loss_fn,C,phi,epochs,test_dataloader = test_dataloader, weights_mask = weights_mask, bias_mask = bias_mask, act_kwargs = act_kwargs, verbose = verbose))]
        if best_test is None or results[-1][1] < best_test[1]  :
            best_test = (it, results[-1][1])
        if best_train is None or results[-1][0] < best_train[1] :
            best_train = (it, results[-1][0])
        all_loss = results[-1][2].evaluate(dataloader,loss_fn,verbose)
        if best_all is None or all_loss < best_all[1] :
            best_all = (it, all_loss)
    return results, best_train, best_test, best_all