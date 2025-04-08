import pandas_utility as PU
import torch

PU.save_WDB([ torch.rand([4,5]), torch.rand([3,2]) ],'./Dump/dump.data')
print(PU.load_WDB('./Dump/dump.data'))