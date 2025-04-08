import torch
import torch.nn as nn
import os.path as path
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from pandas import DataFrame

functionalList = [ 'ReLU', 'Leaky_ReLU', 'ReLU6', 'Hardsigmoid', 'Hardtanh' ]

def randomHiddenArch(L : int, max_C : int) :
    hidden_C = torch.randint(low = 1, high = max_C, size = (L - 1,)).tolist()
    phi = [ functionalList[i] for i in torch.randint(low = 0, high = len(functionalList), size = (L,)) ]
    return (hidden_C, phi)

def hiddenArch(L : int, C_n : int, phi_str : str) :
    hidden_C = [ C_n for _ in range(L) ]
    phi = [ phi_str for _ in range(L) ]
    return (hidden_C, phi)

class CustomModule(nn.Module):
    def append(self, rectifier : str) :
        match rectifier :
            case "ReLU" :
                self.model.append(nn.ReLU())
            case "Leaky_ReLU" :
                self.model.append(nn.PReLU())
            case "ReLU6" :
                self.model.append(nn.ReLU6())
            case "Hardsigmoid" :
                self.model.append(nn.Hardsigmoid())
            case "Hardtanh" :
                self.model.append(nn.Hardtanh(-1,1))
    
    def __init__(self,C : list[int], phi : list[str]) :
        super().__init__()
        self.params = {}
        self.params["C"] = C
        self.params["phi"] = phi
        self.model = nn.Sequential()
        for k in range(len(self.params["C"]) - 1) :
            self.model.append(nn.Linear(self.params["C"][k],self.params["C"][k + 1],True))
            self.append(self.params["phi"][k])

    def forward(self, input_tensor : Tensor) -> Tensor:
        return self.model(input_tensor)

    def train_loop(self, dataloader : DataLoader, loss_fn : (...), optimizer : Optimizer, epochs : int, verbose : bool = False, connection_masks : None | list[Tensor] = None) :
        for _ in range(epochs):
            # Set the model to training mode - important for batch normalization and dropout layers
            # Unnecessary in this situation but added for best practices
            self.model.train()
            for batch, (X, y) in enumerate(dataloader):
                # Compute prediction and loss
                pred = self.model(X)
                loss = loss_fn(pred, y)

                # Backpropagation
                loss.backward()
                if connection_masks is not None :
                    next_mask = iter(connection_masks)
                    for name,module in self.model.named_modules() :
                        if name == 'Linear' :
                            mask = next(next_mask)
                            w_mask, b_mask = mask.tensor_split([1],dim = 2)
                            module.weight.grad *= w_mask
                            module.bias.grad *= b_mask
                optimizer.step()
                optimizer.zero_grad()

                if batch % 100 == 0:
                    loss = loss.item()
                    if verbose :
                        print(f"loss: {loss:>7f}")

    def test_loop(self, dataloader : DataLoader, loss_fn : (...), verbose : bool = False) :
        self.model.eval()
        num_batches = len(dataloader)
        test_loss = 0

        with torch.no_grad() :
            for X, y in dataloader :
                pred = self.model(X)
                test_loss += loss_fn(pred, y).item()

        test_loss /= num_batches
        if verbose :
            print(f"Avg loss: {test_loss:>8f} \n")
        return test_loss

class NN_Architecture :
    def __init__(self,hidden_C : list[int], C_0 : int, C_L : int, phi : list[str]) -> None:
        self.C = [C_0, *hidden_C, C_L]
        self.phi = phi
        self.L = len(self.C) - 1
        self.connections_masks = []
        for k in range(self.L) :
            self.connections_masks += [ torch.ones(self.C[k+1], self.C[k] + 1) ]
    
    def get_neighbors()

class NN_Training_Evaluator :
    def __init__(self,train_dataloader : DataLoader, test_dataloader : DataLoader, C_0 : int, C_L : int, optimizer : (...), loss_fn : (...), epochs : int, verbose : bool = False) :
        self.train_dataloader, self.test_dataloader, self.loss_fn, self.epochs = train_dataloader, test_dataloader, loss_fn, epochs
        self.base_optimizer = optimizer
        self.verbose = verbose
        self.optimizer = None
        self.C_0, self.C_L = C_0, C_L
        self.train_loader = train_dataloader
        self.test_loader = test_dataloader
        self.torch_module = None
    
    def __call__(self, hidden_C : list[int], hidden_phi : list[str]) :
        C = [self.C_0, *hidden_C, self.C_L]
        self.torch_module = CustomModule(C,hidden_phi)
        self.optimizer = self.base_optimizer(self.torch_module.parameters())
        self.torch_module.train_loop(self.train_loader,self.loss_fn,self.optimizer,self.epochs,self.verbose)
        return self.torch_module.test_loop(self.test_loader,self.loss_fn,self.verbose)
        
def read_parameters(arch_path : str) :
    with open(path.normpath(arch_path), mode = 'rt', encoding = 'utf-8') as fo :
        HC = [int(string) for string in fo.readline().strip().split() ]
        phi = [string for string in fo.readline().strip().split() ]
    return (HC, phi)