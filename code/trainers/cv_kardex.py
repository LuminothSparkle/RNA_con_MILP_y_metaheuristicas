import torch, pickle
from pathlib import Path
from torch import optim
from torch.nn.init import calculate_gain
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold, BaseCrossValidator

from utility.kardex.KardexDataset import KardexDataset
from cv import crossvalidation, visualize_results

def crossvalidate(name : str, class_targets : list[str], regression_targets : list[str], ignore_features : list[str], crossvalidator : BaseCrossValidator, seed : int | None = None, noise : bool = False) :
    torch.set_default_dtype(torch.double)
    metrics = {
        'class' : ['accuracy', 'precision', 'recall', 'average precision', 'balanced accuracy', 'roc auc', 'f1 score', 'log loss', 'matthews corrcoef'],
        'class percentages': ['accuracy', 'precision', 'recall', 'average precision', 'balanced accuracy', 'roc auc', 'f1 score'],
        'regression' : ['absolute percentage', 'absolute', 'squared', 'r2 score', 'd2 absolute', 'explained variance'],
        'regression percentages' : ['r2 score', 'd2 absolute', 'explained variance'],
    }
    folder_path = Path('') / 'Data' / 'kardex'
    with torch.device('cuda') :
        dataset = KardexDataset(folder_path / f'{name}.csv', noise = noise, class_targets = class_targets, ignore_features = ignore_features, regression_targets = regression_targets)
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
        'iterations' : 25,
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
    torch.manual_seed(seed)
    cv = crossvalidation(C = hyperparams['C'], dataset = dataset, # type: ignore
                        base_optimizer = hyperparams['optimizer'], opt_kwargs = hyperparams['optimizer kwds'] if 'optimizer kwds' in hyperparams else {}, # type: ignore
                        loss_fn = hyperparams['loss fn'], epochs = hyperparams['epochs'], iterations = hyperparams['iterations'], # type: ignore
                        base_scheduler = hyperparams['scheduler'] if 'scheduler' in hyperparams else None,  # type: ignore
                        sch_kwargs = hyperparams['scheduler kwds'] if 'scheduler kwds' in hyperparams else {}, verbose = True, # type: ignore
                        crossvalidator = hyperparams['crossvalidator'], train_batches = hyperparams['train batches'],  # type: ignore
                        early_tolerance = hyperparams['early tolerance'] if 'early tolerance' in hyperparams else None, hyperparams = hyperparams) # type: ignore
    filepath = folder_path / f'{name}'
    with open(filepath / 'results.pkl','wb') as fo :
        pickle.dump(cv,fo)
    torch.save(cv['model'][0].state_dict(),filepath / 'best.pt')
    onnx_program = torch.onnx.export(cv['model'][0], args = (torch.ones((1,C_0), device = 'cuda', dtype = torch.double),), dynamo = True, export_params = True)
    onnx_program.optimize() # type: ignore
    onnx_program.save( destination = filepath / 'best.onnx') # type: ignore
    #torch.jit.script(cv['model'][0]).save(filepath / 'best_script.pt')
    torch.save(cv['model'][-1].state_dict(),filepath / 'worst.pt')
    onnx_program = torch.onnx.export(model = cv['model'][-1], args = (torch.ones((1,C_0), device = 'cuda', dtype = torch.double),), dynamo = True, export_params = True)
    onnx_program.optimize() # type: ignore
    onnx_program.save( destination = filepath / 'worst.onnx') # type: ignore
    #torch.jit.script(cv['model'][-1]).save(filepath / 'worst_script.pt')
    return cv

def evaluate(name : str) :
    torch.set_default_dtype(torch.double)
    seed = 42
    
    dataset_params = [
#        (['estado'],['trimestre'],[],False,RepeatedStratifiedKFold(n_splits = 10, n_repeats = 10)),
#        (['estado','trimestre'],[],[],False,RepeatedKFold(n_splits = 10, n_repeats = 10)),
        (['estado'],[],['trimestre'],False,RepeatedStratifiedKFold(n_splits = 5, n_repeats = 5, random_state = seed)),
#        ([],['trimestre'],['estado'],False,RepeatedKFold(n_splits = 5, n_repeats = 5, random_state = seed)),
#        (['trimestre'],[],['estado'],False,RepeatedKFold(n_splits = 10, n_repeats = 10))
    ]
    metrics = {
        'class' : ['accuracy', 'precision', 'recall', 'average precision', 'balanced accuracy', 'roc auc', 'f1 score', 'log loss', 'matthews corrcoef'],
        'class percentages': ['accuracy', 'precision', 'recall', 'average precision', 'balanced accuracy', 'roc auc', 'f1 score'],
        'regression' : ['absolute percentage', 'absolute', 'squared', 'r2 score', 'd2 absolute', 'explained variance'],
        'regression percentages' : ['r2 score', 'd2 absolute', 'explained variance'],
    }
    folder_path = Path('') / 'Data' / 'kardex'
    for cls_tgt, reg_tgt, ign_fea, noise, crossvalidator in dataset_params :
        cv = crossvalidate(name,cls_tgt,reg_tgt,ign_fea,crossvalidator,seed,False)
        visualize_results(cv,metrics)

if __name__ == '__main__' :
    for name in ['ambiental 1_15','civil 2_14','computacion 122_9','electrica 3_17','electronica 9_15','fisica 4_16','industrial 5_14','mecanica 6_14','metalurgica 7_14','quimica 8_12'] :
        evaluate(name)