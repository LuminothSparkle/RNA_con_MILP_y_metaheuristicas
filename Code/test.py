import torch
import NN_framework as NNF
import os.path as path
import sys
from pandas import read_csv
from torch.utils.data import DataLoader
from torch.utils.data import BatchSampler
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset
import torch.utils.data as data
import SA_torch as SA

class ArchOptimizer :
    def __init__(self, dataset : Dataset, C_0 : int, C_L : int, optimizer, loss_fn, epochs, num_folds : int = 10, batch_size : int = 32, C_max : int = 10) :
        self.dataset, self.batch_size = dataset, batch_size
        self.T = len(dataset)
        self.indexes = torch.randperm(T)
        self.C_max, self.num_folds = C_max, num_folds
        self.C_0, self.C_L = C_0, C_L
        self.optimizer, self.loss_fn, self.epochs = optimizer, loss_fn, epochs
        
    
    def __call__(self, L : int):
        loss = 0
        hidden_C, phi = NNF.randomHiddenArch(L,C_max)
        for _ in range(self.num_folds) :
            test_indexes, train_indexes = data.random_split(self.indexes,[T / self.num_folds, 1 - T / self.num_folds])
            test_sampler = BatchSampler(test_indexes, batch_size = self.batch_size, drop_last = False)
            train_sampler = BatchSampler(train_indexes, batch_size = self.batch_size, drop_last = False)
            train_dataloader = DataLoader(dataset,train_sampler)
            test_dataloader = DataLoader(dataset,test_sampler)
            trainer = NNF.NN_Training_Evaluator(train_dataloader = train_dataloader, test_dataloader = test_dataloader, C_0 = self.C_0, C_L = self.C_L, optimizer = self.optimizer, loss_fn = self.loss_fn, epochs = self.epochs)
            loss += trainer(hidden_C,phi)
        return loss / self.num_folds
    
tensor_path = path.normpath(sys.argv[2])
label_path = path.normpath(sys.argv[3])

tensor_dataframe = read_csv(tensor_path)
label_dataframe = read_csv(label_path)

C_max = torch.randint(10,30)
T = tensor_dataframe.shape[0]
batch_size = 64
num_folds = 10
indexes = torch.randperm(T)
dataset = TensorDataset( torch.tensor( tensor_dataframe.values() ), torch.tensor( label_dataframe.values() ) )
optimizer = lambda parameters : torch.optim.SGD(parameters, lr = 0.00001)
loss_fn = torch.nn.SmoothL1Loss()
epochs = 1000
L_0 = torch.randint(10,30)
C_max = torch.randint(10,30)
C_0 = tensor_dataframe.shape[1]
C_L = label_dataframe.shape[1]
arch_opt = ArchOptimizer(dataset,C_0,C_L,optimizer,loss_fn,epochs,num_folds,batch_size,C_max)
loss_0 = arch_opt(L_0)
init_temperature = 100
neighbor_fn = lambda L : [L - 1, L + 1]
iterations = 100
L_best, loss_best = SA.simulated_annealing(arch_opt, L_0, loss_0, SA.defaultTemperature, init_temperature, neighbor_fn, SA.defaultProbability, iterations, 'MIN')
print(L_best, loss_best)
