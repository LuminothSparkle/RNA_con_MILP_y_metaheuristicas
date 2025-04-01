import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import collections

class CustomDataset(Dataset):
    def __init__(self,tensor_path,label_path):
        self.tensor_list = []
        with open(tensor_path) as openfileobject:
            for line in openfileobject:
                self.tensor_list += [[float(string) for string in line.strip().split()]]
        self.label_list = []
        with open(label_path) as openfileobject:
            for line in openfileobject:
                self.label_list += [[float(string) for string in line.strip().split()]]

    def __len__(self):
        return len(self.tensor_list)

    def __getitem__(self, idx):
        return torch.Tensor(self.tensor_list[idx]), torch.Tensor(self.label_list[idx])

class Net(nn.Module):
    def __init__(self,input_size):
        super(Net, self).__init__()
        layers = collections.OrderedDict([])
        next_layer_size = input_size
        hidden_layers_size = [32, 32]
        i = 1
        layers["flatten"] = nn.Flatten()
        for layer_size in hidden_layers_size :
            layers["hidden_" + str(i)] = nn.Linear(next_layer_size, layer_size)
            layers["ReLU_" + str(i)] = nn.ReLU()
            next_layer_size = layer_size
            i = i + 1
        layers["output"] = nn.Linear(next_layer_size, 50)
        self.model = nn.Sequential(layers)

    def forward(self, input):
        return self.model(input)

def train_loop(dataloader, model, loss_fn, optimizer) :
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * 64 + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn) :
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad() :
        for X, y in dataloader :
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

training_data = CustomDataset('','')
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
net = Net(training_data.input_size)

epochs = 10
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader,net,loss,optimizer)

