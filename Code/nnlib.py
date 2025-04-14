import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import BatchSampler
from torch.utils.data import SequentialSampler
import torch.utils.data as data
from random import randint
import random

functional_dict : dict[str, nn.Module] = {
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

class NN_Architecture :
    def __init__(self, C : list[int], phi : list[str], masks : None | list[tuple[Tensor,Tensor]] = None) -> None :
        assert len(C) == len(phi) + 1, 'Size of C must be size of phi + 1'
        self.C, self.phi = C, phi
        self.masks, self.L = masks, len(C)
            
    def random_neighbor(self)  -> "NN_Architecture" :
        L = randint(max(self.L - 1,2), self.L + 1)
        C = [ randint(max(c - 1,1), c + 1) for c in self.C[1:-2] ]
        phi = self.phi
        if L < self.L :
            idx = randint(0,len(C) - 1)
            C.pop(idx)
            phi.pop(idx)
        elif L > self.L :
            idx = randint(0, len(C))
            C.insert(idx,randint(1,10))
            phi.insert(idx,random.choice([*functional_dict.keys()]))
        C.insert(0,self.C[0]); C.insert(len(C),self.C[-1])
        return NN_Architecture(C,phi)

class LinealNN(nn.Module):
    def __init__(self,C : list[int], phi : list[str], weights : None | list[tuple[Tensor,Tensor]] = None, mask : None | list[tuple[Tensor,Tensor]] = None, act_opts : list[dict] | None = None) :
        assert len(C) - 1 == len(phi), 'Size of phi must be the size of C - 1'
        super().__init__()
        self.train_loss, self.test_loss = [None] * 2
        self.model, self.linear_layers = nn.Sequential(), []
        self.activation_layers = []
        self.weights, self.bias  = [], []
        for k in range(len(C) - 1) :
            self.linear_layers += [ nn.Linear(C[k], C[k + 1], True) ]
            self.weights += [ self.linear_layers[-1].weight ]; self.bias += [ self.linear_layers[-1].bias ]
            if act_opts is not None :
                self.activation_layers += [ functional_dict[phi[k]](**act_opts[k]) ]
            else :
                self.activation_layers += [ functional_dict[phi[k]]() ]
            self.model.append(self.linear_layers[-1]); self.model.append(self.activation_layers[-1])
            if weights is not None :
                if mask is not None :
                    self.linear_layers[-1].weight = weights[k][0] * mask[k][0]
                    self.linear_layers[-1].bias = weights[k][1] * mask[k][1]
                else :
                    self.linear_layers[-1].weight = weights[k][0]
                    self.linear_layers[-1].bias = weights[k][1]

    def forward(self, input_tensor : Tensor) -> Tensor:
        return self.model(input_tensor)

    def train_loop(self, dataloader : DataLoader, loss_fn : (...), optimizer : Optimizer, epochs : int, verbose : bool = False, mask : None | list[tuple[Tensor,Tensor]] = None) -> Tensor:
        self.model.train()
        self.train_loss = torch.zeros(epochs,len(dataloader))
        for it in range(epochs):
            for batch, (X, y) in enumerate(dataloader) :
                pred = self.model(X)
                loss = loss_fn(pred, y)
                loss.backward()
                if mask is not None :
                    for k, module in enumerate(self.linear_layers) :
                        module.weight.grad *= mask[k][0]
                        module.bias.grad *= mask[k][1]
                optimizer.step(); optimizer.zero_grad()
                self.train_loss[it,batch] = loss.item()
                if verbose :
                    print(f"loss : {loss.item():>7f}")
        return self.train_loss[-1,:].mean()

    def test_loop(self, dataloader : DataLoader, loss_fn : (...), verbose : bool = False) -> Tensor :
        self.model.eval()
        self.test_loss = torch.zeros(len(dataloader))
        with torch.no_grad() :
            for t, (X, y) in enumerate(dataloader) :
                pred = self.model(X)
                self.test_loss[t] = loss_fn(pred, y).item()
        if verbose :
            print(f"Avg loss: {self.test_loss.mean():>8f} \n")
        return self.test_loss.mean()

    def get_weights(self) :
        return [module.weight for module in self.linear_layers], [module.bias for module in self.linear_layers]
    
    def get_PreLU_weights(self) :
        return [ layer.weight if isinstance(layer,nn.PReLU) else None for layer in self.activation_layers ]

class NN_Training_Evaluator :
    def __init__(self, train_dataloader : DataLoader, test_dataloader : DataLoader, optimizer : (...), loss_fn : (...), epochs : int, verbose : bool = False, act_ops : None | list[dict] = None) :
        self.train_dataloader, self.test_dataloader = train_dataloader, test_dataloader
        self.loss_fn, self.epochs = loss_fn, epochs
        self.base_optimizer, self.verbose = optimizer, verbose
        self.train_loader, self.test_loader = train_dataloader, test_dataloader
        self.torch_module, self.optimizer = [None] * 2
        self.train_loss, self.test_loss = [None] * 2
        self.best_loss, self.best_arch, self.best_module = [None] * 3
        self.act_ops = act_ops
    
    def __call__(self, arch : NN_Architecture) :
        self.torch_module = LinealNN(arch.C,arch.phi, act_opts = self.act_ops)
        self.optimizer = self.base_optimizer(self.torch_module.parameters())
        train_loss = self.torch_module.train_loop(self.train_loader,self.loss_fn,self.optimizer,self.epochs,self.verbose, mask = arch.masks)
        test_loss  = self.torch_module.test_loop(self.test_loader,self.loss_fn,self.verbose)
        self.train_loss = self.torch_module.train_loss
        self.test_loss = self.torch_module.test_loss
        if self.verbose :
            print(f"loss: {test_loss:>8f}")
        if self.best_loss is None or test_loss * 0.65 + train_loss * 0.35 < self.best_loss :
            self.best_loss = test_loss * 0.65 + train_loss * 0.35
            self.best_arch = arch
            self.best_module = self.torch_module
        return test_loss * 0.65 + train_loss * 0.35

class ArchEvaluator :
    def __init__(self, train_dataset : Dataset, optimizer : (...), loss_fn : (...), epochs : int = 100, test_dataset : None | Dataset = None, act_ops : None | list[dict] = None, test_percentage : float = 0.3, batch_size : int = 32, iterations : int = 100) :
        self.test_dataset, self.train_dataset = test_dataset, train_dataset
        self.test_percentage, self.iterations = test_percentage, iterations
        self.batch_size, self.epochs = batch_size, epochs
        self.optimizer, self.loss_fn = optimizer, loss_fn
        self.test_loss, self.train_loss = [None] * 2
        self.best_loss, self.best_arch, self.best_module = [None] * 3
        self.act_ops = act_ops
        
    def __call__(self, arch : NN_Architecture):
        if self.test_dataset is not None :
            test_indexes, train_indexes = torch.randperm(len(self.test_dataset)), torch.randperm(len(self.train_dataset))
            test_sampler = SequentialSampler(test_indexes)
            train_sampler = BatchSampler(train_indexes, batch_size = self.batch_size, drop_last = False)
            train_dataloader = DataLoader(self.train_dataset, batch_sampler = train_sampler)
            test_dataloader = DataLoader(self.test_dataset, sampler = test_sampler)
            trainer = NN_Training_Evaluator(train_dataloader = train_dataloader, test_dataloader = test_dataloader, optimizer = self.optimizer, loss_fn = self.loss_fn, epochs = self.epochs, act_ops = self.act_ops)
            loss = trainer(arch)
            self.train_loss, self.test_loss = trainer.train_loss, trainer.test_loss
            if trainer.best_loss is not None and (self.best_loss is None or trainer.best_loss < self.best_loss) :
                self.best_loss = trainer.best_loss
                self.best_arch = trainer.best_arch
                self.best_module = trainer.best_module
        else :
            data_size = len(self.train_dataset)
            loss = torch.zeros(self.iterations)
            test_size = int(data_size * self.test_percentage)
            train_size = data_size - test_size
            train_batches = train_size // self.batch_size + int(train_size % self.batch_size > 0)
            self.train_loss = torch.zeros(self.iterations,self.epochs,train_batches)
            self.test_loss = torch.zeros(self.iterations,test_size)
            for it in range(self.iterations) :
                test_indexes, train_indexes = data.random_split( torch.randperm(data_size), [test_size, train_size] )
                test_sampler = SequentialSampler(test_indexes)
                train_sampler = BatchSampler(train_indexes, batch_size = self.batch_size, drop_last = False)
                train_dataloader = DataLoader(self.train_dataset, batch_sampler = train_sampler)
                test_dataloader = DataLoader(self.train_dataset, sampler = test_sampler)
                
                trainer = NN_Training_Evaluator(train_dataloader = train_dataloader, test_dataloader = test_dataloader, optimizer = self.optimizer, loss_fn = self.loss_fn, epochs = self.epochs, act_ops = self.act_ops)
                loss[it] = trainer(arch)
                self.train_loss[it,...] = trainer.train_loss
                self.train_loss[it,...], self.test_loss[it,...] = trainer.train_loss, trainer.test_loss
                if trainer.best_loss is not None and (self.best_loss is None or trainer.best_loss < self.best_loss) :
                    self.best_loss = trainer.best_loss
                    self.best_arch = trainer.best_arch
                    self.best_module = trainer.best_module
            loss = loss.mean()
        return loss