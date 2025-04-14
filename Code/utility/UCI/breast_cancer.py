from pandas import (Index, DataFrame, read_csv)
import argparse
from argparse import ArgumentParser
from pathlib import (Path, PurePath)

argparser = ArgumentParser()
argparser.add_argument('--save_path', '-sp')
argparser.add_argument('--load_path', '-lp')

def get_dataframes(load_path : PurePath) :
    columns = Index(['Class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat'])
    breast_cancer_frame = read_csv(load_path / 'breast-cancer.data', names = columns)
    indexes = breast_cancer_frame.index

    age_index = Index(['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99'])
    inv_nodes_index = Index(['0-2', '3-5', '6-8', '9-11', '12-14', '15-17', '18-20', '21-23', '24-26', '27-29', '30-32', '33-35', '36-39'])
    tumor_size_index = Index(['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59'])
    menopause_index = Index(['premeno', 'lt40', 'ge40'])
    breast_index = Index(['left', 'right'])
    breast_quad_dict = {'left_up' : [-1,1,0], 'left_low' : [-1,-1,0], 'right_up' : [1,1,0], 'right_low' : [1,-1,0], 'central' : [0,0,1], '?' : [0,0,0]}
    irradiat_index = Index(['no','yes'])
    node_caps_index = Index(['no','yes'])
    Class_index = Index(['no-recurrence-events', 'recurrence-events'])

    feature_columns = ['age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'breast-quad-left-right', 'breast-quad-low-up', 'breast-quad-central', 'irradiat']
    feature_frame = DataFrame(index = indexes, columns = feature_columns, dtype = 'Int64')
    target_frame = DataFrame(index = indexes, columns = ['Class'], dtype = 'Int64')
    for idx in indexes :
        target_frame.at[idx,'Class'] = Class_index.to_list().index(breast_cancer_frame.at[idx,'Class'])
        feature_frame.at[idx,'age'] = age_index.to_list().index(breast_cancer_frame.at[idx,'age']) + 1
        feature_frame.at[idx,'menopause'] = menopause_index.to_list().index(breast_cancer_frame.at[idx,'menopause']) + 1
        feature_frame.at[idx,'tumor-size'] = tumor_size_index.to_list().index(breast_cancer_frame.at[idx,'tumor-size']) + 1
        feature_frame.at[idx,'inv-nodes'] = inv_nodes_index.to_list().index(breast_cancer_frame.at[idx,'inv-nodes']) + 1
        feature_frame.at[idx,'deg-malig'] = breast_cancer_frame.at[idx,'deg-malig']
        feature_frame.at[idx,'breast'] = breast_index.to_list().index(breast_cancer_frame.at[idx,'breast']) * 2 - 1
        if breast_cancer_frame.at[idx,'node-caps'] == '?' :
            feature_frame.at[idx,'node-caps'] = 0
        else :
            feature_frame.at[idx,'node-caps'] = node_caps_index.to_list().index(breast_cancer_frame.at[idx,'node-caps']) * 2 - 1
        feature_frame.at[idx,'irradiat'] = irradiat_index.to_list().index(breast_cancer_frame.at[idx,'irradiat']) * 2 -1
        feature_frame.loc[idx,['breast-quad-left-right','breast-quad-low-up','breast-quad-central']] = breast_quad_dict[breast_cancer_frame.at[idx,'breast-quad']]
    return feature_frame, target_frame

def save_dataframes(feature_frame : DataFrame, target_frame : DataFrame, save_path : PurePath) :
    feature_frame.to_csv(save_path / 'features.csv',index_label = 'Case')
    target_frame.to_csv(save_path / 'target.csv',index_label = 'Case')

def main(args : argparse.Namespace) :
    if args.load_path is not None :
        load_path = Path(args.load_path)
    else :
        load_path = Path('')
    if args.save_path is not None :
        save_path = Path(args.save_path)
    else :
        save_path = Path(load_path)
    if not save_path.exists() :
        print(f'{save_path} doesn\'t exists')
        return None
    elif not save_path.is_dir() :
        print(f'Cannot access {save_path} or isn\'t a directory')
        return None       
    if not load_path.exists() :
        print(f'{load_path} doesn\'t exists')
        return None
    elif not load_path.is_dir() :
        print(f'Cannot access {load_path} or isn\'t a directory')
        return None
    if (save_path / 'features.csv').exists() or (save_path / 'targets.csv').exists() :
        print('features.csv or targets.csv already exists')
        yes_list = ['y','YES','yes','Y']; no_list = ['n','NO','no','N']
        conf = input('Overwrite : [yes/no]')
        while conf not in yes_list and conf not in no_list :
            print(f'Response must be : {", ".join(yes_list)}, {", ".join(no_list)}')
            conf = input('Overwrite : [yes/no]')
        if conf in no_list :
            return None
    feature_frame, target_frame = get_dataframes(load_path)
    save_dataframes(feature_frame,target_frame,save_path)

if __name__ == '__main__' :
    import sys
    sys.exit(main(argparser.parse_args(sys.argv)))
