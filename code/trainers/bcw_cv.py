import torch, pickle
from pathlib import Path
from torch import optim
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold
from torch.nn.init import calculate_gain

from cv import crossvalidation, visualize_results
from code.utility.UCI.BCWDataset import BCWDataset


if __name__ == '__main__' :
    metrics = {
        'class' : ['accuracy', 'precision', 'recall', 'average precision', 'balanced accuracy', 'roc auc', 'f1 score', 'log loss', 'matthews corrcoef'],
        'class percentages': ['accuracy', 'precision', 'recall', 'average precision', 'balanced accuracy', 'roc auc', 'f1 score'],
        'regression' : ['absolute percentage', 'absolute', 'squared', 'r2 score', 'd2 absolute'],
        'regression percentages' : ['r2 score', 'd2 absolute', 'explained variance'],
    }
    seed = 42
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
        'iterations' : 25,
#        'epochs' : 10000,
        'train batches' : 100,
#        'train batches' : len(dataset) // 32,
        'epochs' : 10000,
#        'early tolerance' : 100,
        'early tolerance' : 50,
#        'crossvalidator' : StratifiedKFold(n_splits = 10, shuffle = True, random_state = 42),
        'crossvalidator' : RepeatedStratifiedKFold(n_splits = 5, n_repeats = 5, random_state = seed),
        'loss fn' : loss_fn,
        'optimizer' : optim.Adam,
#        'optimizer' : optim.AdamW,
        'optimizer kwds' : {'fused' : True },
#        'optimizer kwds' : {'fused' : True, 'lr' : 0.01  },
        'scheduler' : optim.lr_scheduler.CyclicLR,
        'scheduler kwds' : {'base_lr' : 0.001, 'max_lr' : 0.01, 'step_size_up' : 1, 'step_size_down' : 10, 'mode' : 'exp_range', 'gamma' : 0.9 },
    }
    torch.manual_seed(seed)    
    cv = crossvalidation(C = hyperparams['C'], dataset = dataset, # type: ignore
                              base_optimizer = hyperparams['optimizer'], opt_kwargs = hyperparams['optimizer kwds'] if 'optimizer kwds' in hyperparams else {}, # type: ignore
                              loss_fn = hyperparams['loss fn'], epochs = hyperparams['epochs'], iterations = hyperparams['iterations'], # type: ignore
                              base_scheduler = hyperparams['scheduler'] if 'scheduler' in hyperparams else None,  # type: ignore
                              sch_kwargs = hyperparams['scheduler kwds'] if 'scheduler kwds' in hyperparams else {}, verbose = True, # type: ignore
                              crossvalidator = hyperparams['crossvalidator'], train_batches = hyperparams['train batches'],  # type: ignore
                              early_tolerance = hyperparams['early tolerance'] if 'early tolerance' in hyperparams else None, hyperparams = hyperparams) # type: ignore
    filepath = Path('') / 'Data' / 'Breast Cancer Winsconsin (Diagnostic)'
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
    visualize_results(cv,metrics)
    