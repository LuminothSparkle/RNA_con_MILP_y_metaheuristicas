import torch
import torch.nn as nn
import torch.optim as optim
import os.path as path
from torch import Tensor
import sys
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self,tensor_path,label_path):
        self.tensor_list = []
        with open(path.normpath(tensor_path), mode = 'rt', encoding = 'utf-8') as openfileobject:
            for line in openfileobject:
                self.tensor_list += [[float(string) for string in line.strip().split()]]
        self.label_list = []
        with open(path.normpath(label_path), mode = 'rt', encoding = 'utf-8') as openfileobject:
            for line in openfileobject:
                self.label_list += [[float(string) for string in line.strip().split()]]

    def __len__(self):
        return len(self.tensor_list)

    def __getitem__(self, idx):
        return Tensor(self.tensor_list[idx]), Tensor(self.label_list[idx])

class CustomModule(nn.Module):
    def __init__(self,arch_path):
        super(CustomModule, self).__init__()
        self.params = nn.ParameterDict()
        with open(path.normpath(arch_path), mode = 'rt', encoding = 'utf-8') as openfileobject:
            self.params["L"] = int(openfileobject.readline().strip())
            self.params["C"] = [int(string) for string in openfileobject.readline().split()]
            self.params["phi"] = [string for string in openfileobject.readline().split()]
        
        self.model = nn.Sequential()
        for k in range(self.params["L"]) :
            self.model.append(nn.Linear(self.params["C"][k],self.params["C"][k + 1],True))
            match self.params["phi"][k] :
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

    def forward(self, input_tensor):
        return self.model(input_tensor)
    
    def train_epoch(self, dataloader, loss_fn, optimizer) :
        size = len(dataloader.dataset)
        # Set the model to training mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        self.model.train()
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            pred = self.model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * 64 + len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def train_loop(self, dataloader, loss_fn, optimizer, epochs) :
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            self.train_epoch(dataloader,loss_fn,optimizer)

    def test_loop(self, dataloader, loss_fn) :
        self.model.eval()
        #size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss = 0
        #correct = 0

        with torch.no_grad() :
            for X, y in dataloader :
                pred = self.model(X)
                test_loss += loss_fn(pred, y).item()
                #correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        #correct /= size
        #print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    def save_parameters(self, save_path) :
        torch.save(self.model.state_dict(),path.normpath(save_path))

    def load_parameters(self, load_path) :
        self.model.load_state_dict(torch.load(path.normpath(load_path), weights_only = True))
        self.model.eval()

    def save_script(self, save_path) :
        model_scripted = torch.jit.script(self.model) # Export to TorchScript
        model_scripted.save(path.normpath(save_path))

    def load_script(self, load_path) :
        self.model = torch.jit.load(path.normpath(load_path))
        self.model.eval()

def save_optimizer_checkpoint(optimizer, save_path) :
    torch.save(optimizer,path.normpath(save_path))

def load_optimizer_checkpoint(optimizer, load_path) :
    optimizer.load_state_dict(torch.load(path.normpath(load_path), weights_only = True))

database_path = path.normpath(sys.argv[1])
labels_path = path.normpath(sys.argv[2])
model_path = path.normpath(sys.argv[3])
saving_path = path.normpath(sys.argv[4])
training_data = CustomDataset(database_path,labels_path)
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
net = CustomModule(model_path)
# Training parameters
epochs_n = 10
loss_function = nn.CrossEntropyLoss()
optim = optim.SGD(net.parameters(), lr = 0.001)
net.train_loop(train_dataloader,loss_function,optim,epochs_n)
net.test_loop(train_dataloader,loss_function)
net.save_parameters(path.join(saving_path, 'model.pth'))
save_optimizer_checkpoint(optim,path.join(saving_path, 'optimizer.pth'))