from collections.abc import Iterable

from pandas import read_csv, Index

from pathlib import Path

import torch
from torch import Tensor
from torch.nn.functional import cross_entropy, smooth_l1_loss, softmax, mse_loss, sigmoid, binary_cross_entropy_with_logits, l1_loss

from cv import CrossvalidationTensorDataset

class KardexDataset(CrossvalidationTensorDataset) :
    noise : bool
        
    def __init__(self, kardex_csv_path : Path, class_targets : str | Iterable[str] | None = 'estado', regression_targets : str | Iterable[str] | None = 'ultimo trimestre activo', ignore_features : Iterable[str] | None = None, noise : bool = False) -> None:
        self.noise = noise
        dataframe = read_csv(filepath_or_buffer = Path(kardex_csv_path), header = 0, index_col = 0)
        labels = {}
        labels['class targets'] = Index(([] if class_targets is None else [class_targets] if isinstance(class_targets, str) else [class_target for class_target in class_targets]))
        labels['regression targets'] = Index(([] if regression_targets is None else [regression_targets] if isinstance(regression_targets, str) else [regression_target for regression_target in regression_targets]))
        labels['ignore'] =  Index([] if ignore_features is None else [ignore_features] if isinstance(ignore_features, str) else [ignore_feature for ignore_feature in ignore_features])
        labels['targets'] = labels['class targets'].union(labels['regression targets'] )
        labels['features'] = dataframe.columns.drop( labels = labels['targets'], errors = 'ignore')
        if ignore_features is not None :
            labels['features'] = labels['features'].drop( labels = labels['ignore'], errors = 'ignore')
        super().__init__(dataframe = dataframe,labels = labels)
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
            elif pred.size(dim = 1) == 1 and target.size(dim = 1) == 1 : 
                loss += binary_cross_entropy_with_logits(pred,target, reduction = 'mean', pos_weight = self.tensors['weights'][label]) # type: ignore
            else :
                loss += cross_entropy(pred,target, reduction = 'mean', weight = self.tensors['weights'][label]) # type: ignore
        return loss / len(list)