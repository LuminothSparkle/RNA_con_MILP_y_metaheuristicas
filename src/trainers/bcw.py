"""
Codigo que entrena la red neuronal de Breast Cancer Winsconsin Diagnostic
"""
import argparse
from argparse import ArgumentParser

from pathlib import Path

import torch

from src.utility.datasets import BCWDataset
from src.utility.nn.crossvalidation import crossvalidate

from src.utility.files import (
    read_arch_json, read_cv_json,
    save_crossvalidation
)

#        'C' : [C_0, 1024, 1024, 512, 512, 256, 256, 256, 256, 256, 256, 256, 256, C_L],
#            ('ReLU',{}),
#            ('ReLU',{}),
#            ('ReLU',{}),
#            ('ReLU',{}),
#            ('ReLU',{}),
#            ('ReLU',{}),
#            ('ReLU',{}),
#            ('ReLU',{}),
#            ('ReLU',{}),
#            ('ReLU',{}),
#            ('ReLU',{}),
#            ('ReLU',{}),
#            ('None',{}),
#        ],
#        'l1 weight regularization' : None,
#        'l2 activation regularization' : 0.000001,
#        'l2 weight regularization' : None,
#        'batch normalization' : tuple({}),
#        'batch normalization' : None,
#        'dropout' : None,
#        'iterations' : 30,
#        'epochs' : 10000,
#        'train batches' : len(dataset) // 32,
#        'early tolerance' : 100,
#        'crossvalidator' : StratifiedKFold(n_splits = 10, shuffle = True, random_state = 42),
#        'optimizer' : optim.AdamW,
#        'optimizer kwds' : {'fused' : True, 'lr' : 0.01  },

def test_arch(dataset : BCWDataset, arch_data : dict, cv_data : dict) :
    """
    A
    """
    torch.manual_seed(arch_data['torch seed'])
    if torch.cuda.is_available() :
        torch.cuda.manual_seed(arch_data['torch seed'])
    loss_fn = dataset.loss_fn
    base_crossvalidator, cv_kwargs = cv_data['crossvalidator']
    crossvalidator = base_crossvalidator(**cv_kwargs)
    return crossvalidate(
        arch = arch_data['capacity'],
        dataset = dataset,
        optimizer = arch_data['optimizer'],
        loss_fn = loss_fn,
        epochs = arch_data['epochs'],
        iterations = cv_data['iterations'],
        crossvalidator = crossvalidator,
        train_batches = cv_data['train batches'],
        extra_params = arch_data
    )

def main(args : argparse.Namespace) :
    """
    Funcion main para generar la base de datos en archivo csv del archivo
    json del kardex anonimizado
    """
    load_path = Path(args.load_path)
    save_path = Path(args.save_path)
    arch_path = Path(args.arch_path)
    cv_path = Path(args.cv_path)
    if not load_path.exists() :
        print(f'{load_path} doesn\'t exists')
        return None
    elif not load_path.is_dir() :
        print(f'Cannot access {load_path} or isn\'t a directory')
        return None
    cv_data = read_cv_json(cv_path)
    arch_data = read_arch_json(arch_path)
    dataset = BCWDataset(cv_data)
    arch_data['capacity'] = [
        dataset.features_size,
        *arch_data['capacity'],
        dataset.targets_size
    ]
    results_dict = test_arch(dataset,arch_data,cv_data)
    results_path = save_path / 'results'
    results_path.mkdir(parents = True, exist_ok = True)
    save_crossvalidation(results_path,results_dict,'bcw',not args.no_overwrite)

if __name__ == '__main__' :
    import sys
    argparser = ArgumentParser()
    argparser.add_argument(
        '--save_path', '-sp',
        default = Path.cwd() / 'Data' /
        'Breast Cancer Winsconsin (Diagnostic)'
    )
    argparser.add_argument(
        '--load_path', '-lp',
        default = Path.cwd() / 'Data' /
        'Breast Cancer Winsconsin (Diagnostic)'
    )
    argparser.add_argument(
        '--arch_path', '-ap',
        default = Path.cwd() / 'Data' /
            'Breast Cancer Winsconsin (Diagnostic)' /
            'crossvalidation' / 'arch.json'
    )
    argparser.add_argument(
        '--cv_path', '-cp',
        default = Path.cwd() / 'Data' /
            'Breast Cancer Winsconsin (Diagnostic)' /
            'crossvalidation' / 'cv.json'
    )
    argparser.add_argument('--no_overwrite', '-no', action = 'store_true')
    torch.set_default_device('cpu')
    torch.set_default_dtype(torch.double)
    if torch.cuda.is_available() :
        torch.set_default_device('cuda')
        torch.set_default_dtype(torch.double)
    sys.exit(main(argparser.parse_args(sys.argv[1:])))
