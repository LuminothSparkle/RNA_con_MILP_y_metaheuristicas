from ucimlrepo import fetch_ucirepo, dotdict

from pandas import Index, read_csv
from pathlib import Path

import torch
from torch import Tensor
from torch.nn.functional import cross_entropy, smooth_l1_loss, mse_loss, binary_cross_entropy_with_logits, l1_loss

from cv import CrossvalidationTensorDataset

class BCWDataset(CrossvalidationTensorDataset) :
    noise : bool
    uci_dataset : dotdict
        
    def __init__(self, noise : bool = False, BCW_file_path : Path | None = None) -> None:
        self.noise = noise
        labels = {'class targets' : Index(['Diagnosis'], dtype = 'string'),
            'regression targets' : Index([]),
            'ignore' : Index(['ID'], dtype = 'string'),
            'features' : Index([f'{s}{idx}' for idx in range(1,4) for s in ['radius','texture','perimeter','area','smoothness','compactness','concavity','concave_points','symmetry','fractal_dimension'] ])
            }
        if BCW_file_path is not None :
            dataframe = read_csv(BCW_file_path, header = None, index_col = 0, names = [*labels['ignore'], *labels['features'], *labels['class targets']])
        else :
            self.uci_dataset = fetch_ucirepo(id = 17)
            dataframe = self.uci_dataset.data.original # type: ignore
        super().__init__(dataframe, labels,True)
        self.sigma = self.tensors['features'].var(dim = 0) # type: ignore
        self.mu = self.tensors['features'].mean(dim = 0) # type: ignore
    
    def loss_fn(self, pred : Tensor, target : Tensor)  -> Tensor:
        target_splitted = self.split_pred_tensor(target)
        pred_splitted = self.split_pred_tensor(pred)
        loss = torch.tensor(0, device = 'cuda', dtype = torch.double)
        list = [*zip(pred_splitted.items(),target_splitted.values())]
        for (label,pred),target in list :
            if label in self.labels['regression targets'] :
                loss += l1_loss(pred,target, reduction = 'mean')
            elif pred.size(dim = 1) > 1 : 
                loss += cross_entropy(pred,target, reduction = 'mean', weight = self.tensors['weights'][label]) # type: ignore
            else :
                loss += binary_cross_entropy_with_logits(pred,target, reduction = 'mean', pos_weight = self.tensors['weights'][label]) # type: ignore
        return loss / len(list)
