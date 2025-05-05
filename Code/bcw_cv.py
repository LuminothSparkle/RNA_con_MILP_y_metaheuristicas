import torch
from pathlib import Path
from torch import optim
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold, StratifiedKFold
from torch.nn.init import calculate_gain

from utility.UCI.BCWDataset import BCWDataset
from cv import crossvalidation_metrics

if __name__ == '__main__' :
    metrics = {
        'class' : ['accuracy', 'precision', 'recall', 'average precision', 'balanced accuracy', 'roc auc', 'f1 score', 'log loss', 'matthews corrcoef'],
        'class percentages': ['accuracy', 'precision', 'recall', 'average precision', 'balanced accuracy', 'roc auc', 'f1 score'],
        'regression' : ['absolute percentage', 'absolute', 'squared', 'r2 score', 'd2 absolute'],
        'regression percentages' : ['r2 score', 'd2 absolute', 'explained variance'],
    }
    torch.set_default_dtype(torch.double)
    with torch.device('cuda') :
        dataset = BCWDataset(noise = False)
        loss_fn = dataset.loss_fn
    C_0 = dataset.size_features()
    C_L = dataset.size_targets()
    hyperparams = {
#        'C' : [C_0, 1024, 1024, 512, 512, 256, 256, 256, 256, 256, 256, 256, 256, C_L],
        'C' : [C_0, 256, 128, 64, 16, C_L],
        'activation function' : [
#            ('ReLU',{}),
#            ('ReLU',{}),
#            ('ReLU',{}),
#            ('ReLU',{}),
#            ('ReLU',{}),
#            ('ReLU',{}),
#            ('ReLU',{}),
#            ('ReLU',{}),
            ('ReLU',{}),
            ('ReLU',{}),
            ('ReLU',{}),
            ('ReLU',{}),
            ('None',{}),
        ],
        'l1 activation regularization' : 0.0005,
#        'l1 weight regularization' : None,
#        'l2 activation regularization' : 0.000001,
#        'l2 weight regularization' : None,
#        'batch normalization' : tuple({}),
#        'batch normalization' : None,
#        'dropout' : None,
        'dropout' : {2 : 0.25, 3 : 0.25, 4 : 0.25},
        'connect dropout' : None,
        'bias init' : torch.ones,
        'weight init' : ('xavier uniform',{'gain' : calculate_gain('relu')}),
#        'iterations' : 30,
        'iterations' : 100,
#        'epochs' : 10000,
        'train batches' : 100,
#        'train batches' : len(dataset) // 32,
        'epochs' : 10000,
#        'early tolerance' : 100,
        'early tolerance' : 50,
#        'crossvalidator' : StratifiedKFold(n_splits = 10, shuffle = True, random_state = 42),
        'crossvalidator' : RepeatedStratifiedKFold(n_splits = 10, n_repeats = 3),
        'loss fn' : loss_fn,
        'optimizer' : optim.Adam,
#        'optimizer' : optim.AdamW,
        'optimizer kwds' : {'fused' : True },
#        'optimizer kwds' : {'fused' : True, 'lr' : 0.01  },
        'scheduler' : optim.lr_scheduler.CyclicLR,
        'scheduler kwds' : {'base_lr' : 0.001, 'max_lr' : 0.01, 'step_size_up' : 1, 'step_size_down' : 10, 'mode' : 'exp_range', 'gamma' : 0.9 },
    }
    
    crossvalidation_metrics(dataset = dataset, hyperparams = hyperparams, metrics = metrics)