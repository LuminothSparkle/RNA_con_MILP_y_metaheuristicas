import torch
import sys
from pathlib import Path
from time import perf_counter_ns
from torch import optim
from utility.kardex.KardexDataset import KardexDataset
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold
import matplotlib.pyplot as pyplot
import numpy
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay, PrecisionRecallDisplay, PredictionErrorDisplay
from numpy import array
from torch.nn.init import calculate_gain
from cv import crossvalidation, crossvalidation_metrics

if __name__ == '__main__' :
    torch.set_default_dtype(torch.double)
    
    dataset_params = [
#        (['estado'],['trimestre'],[],False,RepeatedStratifiedKFold(n_splits = 10, n_repeats = 10)),
#        (['estado','trimestre'],[],[],False,RepeatedKFold(n_splits = 10, n_repeats = 10)),
#        (['estado'],[],['trimestre'],False,RepeatedStratifiedKFold(n_splits = 10, n_repeats = 3)),
        ([],['trimestre'],['estado'],False,RepeatedKFold(n_splits = 10, n_repeats = 3)),
#        (['trimestre'],[],['estado'],False,RepeatedKFold(n_splits = 10, n_repeats = 10))
    ]
    metrics = {
        'class' : ['accuracy', 'precision', 'recall', 'average precision', 'balanced accuracy', 'roc auc', 'f1 score', 'log loss', 'matthews corrcoef'],
        'class percentages': ['accuracy', 'precision', 'recall', 'average precision', 'balanced accuracy', 'roc auc', 'f1 score'],
        'regression' : ['absolute percentage', 'absolute', 'squared', 'r2 score', 'd2 absolute', 'explained variance'],
        'regression percentages' : ['r2 score', 'd2 absolute', 'explained variance'],
    }
    for cls_tgt, reg_tgt, ign_fea, noise, crossvalidator in dataset_params :
        with torch.device('cuda') :
            dataset = KardexDataset(Path('./Data/kardex/computacion 122_9.csv'), noise = noise, class_targets = cls_tgt, ignore_features = ign_fea, regression_targets = reg_tgt)
        loss_fn = dataset.loss_fn
        C_0 = dataset.size_features()
        C_L = dataset.size_targets()
        hyperparams = {
#            'C' : [C_0, 256, 64, 16, C_L],
            'C' : [C_0, 256, 64, 64, 64, 64, C_L],
#            'activation function' : [
#                ('LeakyReLU',{'negative_slope' : 0.0000000001}),
#                ('LeakyReLU',{'negative_slope' : 0.00000001}),
#                ('LeakyReLU',{'negative_slope' : 0.000001}),
#                ('LeakyReLU',{'negative_slope' : 10}),
#            ],
            'activation function' : [
                ('LeakyReLU',{'negative_slope' : 0.01}),
                ('LeakyReLU',{'negative_slope' : 0.01}),
                ('LeakyReLU',{'negative_slope' : 0.01}),
                ('LeakyReLU',{'negative_slope' : 0.01}),
                ('LeakyReLU',{'negative_slope' : 0.01}),
                ('LeakyReLU',{'negative_slope' : 0.01}),
            ],
#            'l1 activation regularization' : None,
#            'l1 weight regularization' : None,
#            'l2 activation regularization' : None,
#            'l2 weight regularization' : None,
            'dropout' : {0 : 0.5},
#            'dropout' : {0 : 0.2},
#            'connection dropout' : 0.1,
#            'batch normalization' : None,
            'bias init' : torch.ones,
#            'weight init' : ('xavier uniform',{'gain' : calculate_gain('leaky_relu',10)}),
            'weight init' : ('xavier uniform',{'gain' : calculate_gain('leaky_relu',0.01)}),
            'iterations' : 30,
            'epochs' : 10000,
#            'train batches' : 16,
            'train batches' : 32,
            'early tolerance' : 300,
            'crossvalidator' : crossvalidator,
            'loss fn' : loss_fn,
            'optimizer' : optim.Adam,
            'optimizer kwds' : {'fused' : True, 'lr' : 0.005  },
            'scheduler' : optim.lr_scheduler.CyclicLR,
#            'scheduler kwds' : {'base_lr' : 0.00001, 'max_lr' : 0.005, 'step_size_up' : 1, 'step_size_down' : 10, 'mode' : 'exp_range', 'gamma' : 0.005 },
            'scheduler kwds' : {'base_lr' : 0.00001, 'max_lr' : 0.005, 'step_size_up' : 1, 'step_size_down' : 10, 'mode' : 'exp_range', 'gamma' : 0.05 },
        }
        crossvalidation_metrics(dataset = dataset, hyperparams = hyperparams, metrics = metrics)