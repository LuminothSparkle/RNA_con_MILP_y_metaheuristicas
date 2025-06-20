"""
Codigo que entrena las redes neuronales del problema del problema de
predecir si un estudiante egresará o abandonará de acuerdo a su historial
académico
"""
import argparse
from argparse import ArgumentParser

from pathlib import Path

import torch

from src.utility.datasets import KardexDataset
from src.utility.nn.crossvalidation import crossvalidate

from src.utility.files import (
    read_arch_json, read_cv_json,
    save_crossvalidation
)

def test_arch(dataset : KardexDataset, arch_data : dict, cv_data : dict) :
    """
    A
    """
    torch.manual_seed(arch_data['torch seed'])
    loss_fn = dataset.loss_fn
    base_cv,cv_kwargs = cv_data['crossvalidator']
    cv = base_cv(**cv_kwargs)
    return crossvalidate(
        arch = arch_data['capacity'],
        dataset = dataset,
        optimizer = arch_data['optimizer'],
        loss_fn = loss_fn,
        epochs = arch_data['epochs'],
        iterations = cv_data['iterations'],
        crossvalidator = cv,
        train_batches = cv_data['train batches'],
        extra_params = arch_data
    )

#        (['estado'],['trimestre'],[],False,RepeatedStratifiedKFold(n_splits = 10, n_repeats = 10)),
#        (['estado','trimestre'],[],[],False,RepeatedKFold(n_splits = 10, n_repeats = 10)),
#        ([],['trimestre'],['estado'],False,RepeatedKFold(n_splits = 5, n_repeats = 5, random_state = seed)),
#        (['trimestre'],[],['estado'],False,RepeatedKFold(n_splits = 10, n_repeats = 10))
#            'C' : [C_0, 256, 64, 16, C_L],
#            'activation function' : [
#                ('LeakyReLU',{'negative_slope' : 0.0000000001}),
#                ('LeakyReLU',{'negative_slope' : 0.00000001}),
#                ('LeakyReLU',{'negative_slope' : 0.000001}),
#                ('LeakyReLU',{'negative_slope' : 10}),
#            ],
#            'l1 activation regularization' : None,
#            'l1 weight regularization' : None,
#            'l2 activation regularization' : None,
#            'l2 weight regularization' : None,
#            'dropout' : {0 : 0.2},
#            'connection dropout' : 0.1,
#            'batch normalization' : None,
#            'weight init' : ('xavier uniform',{'gain' : calculate_gain('leaky_relu',10)}),
#            'train batches' : 16,
#            'scheduler kwds' : {'base_lr' : 0.00001, 'max_lr' : 0.005, 'step_size_up' : 1, 'step_size_down' : 10, 'mode' : 'exp_range', 'gamma' : 0.005 },

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
    for career_path in load_path.glob('*.csv') :
        print(f'Processing {career_path.stem}')
        dataset = KardexDataset(career_path,cv_data)
        arch_data['capacity'] = [
            dataset.features_size,
            *arch_data['capacity'],
            dataset.targets_size
        ]
        data = test_arch(dataset,arch_data,cv_data)
        name = career_path.stem
        results_path = save_path / name / 'results'
        results_path.mkdir(parents = True, exist_ok = True)
        save_crossvalidation(results_path,data,name,not args.no_overwrite)

if __name__ == '__main__' :
    import sys
    argparser = ArgumentParser()
    argparser.add_argument('--save_path', '-sp', default = Path.cwd() / 'Data' / 'kardex')
    argparser.add_argument('--load_path', '-lp', default = Path.cwd() / 'Data' / 'kardex')
    argparser.add_argument(
        '--arch_path', '-ap',
        default = Path.cwd() / 'Data' / 'kardex' / 'crossvalidation' / 'arch.json'
    )
    argparser.add_argument(
        '--cv_path', '-cp',
        default = Path.cwd() / 'Data' / 'kardex' / 'crossvalidation' / 'cv.json'
    )
    argparser.add_argument('--no_overwrite', '-no', action = 'store_true')
    torch.set_default_device('cpu')
    torch.set_default_dtype(torch.double)
    if torch.cuda.is_available() :
        torch.set_default_device('cuda')
        torch.set_default_dtype(torch.double)
    sys.exit(main(argparser.parse_args(sys.argv[1:])))
