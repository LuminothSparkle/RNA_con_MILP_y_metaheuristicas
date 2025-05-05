import json
from pandas import (DataFrame, read_csv)
from pathlib import Path
import argparse
from argparse import ArgumentParser
import pandas
import numpy

argparser = ArgumentParser()
argparser.add_argument('--save_path', '-sp')
argparser.add_argument('--load_path', '-lp')
argparser.add_argument('--no_overwrite', '-no', action = 'store_true')
argparser.add_argument('--gen_database', '-gd', action = 'store_true')

calificacion_dict = {
    "RE" : 8,
    "AO" : 8,
    "EE" : 8,
    "EQ" : 8,
    "MB" : 10,
    "B" : 8,
    "S" : 6,
    "NA" : 4
}
order = 'IPO'
len_order = len(order)

def Tri2Int(code : str)  -> int:
    return int(code[0:2]) * len_order + order.index(code[2])

def Int2Tri(num : int)  -> str:
    return f'{num // len_order:02d}{order[num % len_order]}'

def write_DB(data_frames : dict[str,DataFrame], save_path : Path, overwrite : bool = True) :
    file_paths = []
    for name in data_frames :
        file_paths += [save_path / f'{name}.csv']
        if file_paths[-1].exists() and not overwrite :
            return False
    for file_path, data_frame in zip(file_paths, data_frames.values()) :
        data_frame.to_csv(file_path, index_label = 'Alumno')
    return True

def separate_targets_and_features(load_path : Path, save_path : Path, overwrite : bool = True) :
    path_list = []
    for file in load_path.glob('*.csv') :
        if file.is_file() :
            save_dir = save_path / file.stem
            targets_path = save_dir / 'targets.csv'
            features_path = save_dir / 'features.csv'
            save_dir.mkdir(parents = True, exist_ok = True)
            if (targets_path.exists() or features_path.exists()) and not overwrite :
                return False
            path_list += [(file, features_path, targets_path)]
    for load_file, features_path, targets_path in path_list :
        career_frame = read_csv(load_file, header = 0, index_col = 0)
        target_frame = career_frame.iloc[:,:3]
        features_frame = career_frame.iloc[:,3:]
        target_frame.to_csv(targets_path)
        features_frame.to_csv(features_path)
    return True

def process_kardex(load_path : Path) :
    plans_dir = load_path / 'Plans'
    assert plans_dir.exists(), f'{plans_dir} doens\'t exist'
    kardex_file = load_path / 'kardex.json'
    with kardex_file.open(mode = 'rt', encoding = 'utf-8') as text_data :
        kardex_dict = json.load(text_data)
    career_dict = {}
    for name, students in kardex_dict.items() :
        career_data = {}
        _,code = name.strip().split()
        career_path = plans_dir / f'{code}.json'
        with career_path.open(mode = 'rt', encoding = 'utf-8') as text_data :
            career_data = json.load(text_data)
        dataframe_list = []
        for student in students :
            student_dict = {'estado' : '', 'trimestre' : 0, 'creditos obligatorios' : 0, 'creditos optativos' : 0}
            trimesters = [ Tri2Int(intent['trimestre']) for intents in student['ueas'].values() for intent in intents ]
            min_tri = numpy.min(trimesters)
            max_tri = numpy.max(trimesters)
            innactivity_tri = Tri2Int('25I') - max_tri + 1
            if (student['estado']  == 'egreso' or student['estado']  == 'baja' or student['estado']  == 'activo' and innactivity_tri > 6) :
                student_dict['trimestre'] = max_tri - min_tri + 1
                student_dict['estado'] = student['estado'] if student['estado']  == 'egreso' or student['estado']  == 'baja' else 'baja' # type: ignore
                for code in career_data['ueas'] :
                    if code in student['ueas'] :
                        if any(intent['calificacion'] != 'NA' for intent in student['ueas'][code]) :
                            if career_data['ueas'][code]['obligatoria'] :
                                student_dict['creditos obligatorios'] += career_data['ueas'][code]['creditos']
                            else :
                                student_dict['creditos optativos'] += career_data['ueas'][code]['creditos']
                        for intent in student['ueas'][code] :
                            student_dict[f'{code}_ini'] = numpy.min( [Tri2Int(intent['trimestre']) - min_tri + 1 for intent in student['ueas'][code]] )
                            student_dict[f'{code}_las'] = numpy.max( [Tri2Int(intent['trimestre']) - min_tri + 1 for intent in student['ueas'][code]] )
                            student_dict[f'{code}_cal'] = numpy.mean( [calificacion_dict[intent['calificacion']] for intent in student['ueas'][code]] ) # type: ignore
                            student_dict[f'{code}_eva'] = numpy.mean( [int(intent['evaluacion'] == 'GLO.') for intent in student['ueas'][code]] ) # type: ignore
                    else :
                        student_dict[f'{code}_ini'] = 0
                        student_dict[f'{code}_las'] = 0
                        student_dict[f'{code}_cal'] = 0
                        student_dict[f'{code}_eva'] = 0
                dataframe_list += [DataFrame(student_dict, index = [0])]
        career_dict[name] = pandas.concat(dataframe_list, axis = 'index', join = 'outer', ignore_index = True).fillna(0)
    return career_dict

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
    if args.gen_database :
        if not (load_path / 'Plans').exists() :
            print(f'{load_path / "Plans"} doesn\'t exists')
            return None
        career_dict = process_kardex(load_path)
        write_DB(career_dict, save_path, not args.no_overwrite)
    else :
        separate_targets_and_features(load_path, save_path,not args.no_overwrite)

if __name__ == '__main__' :
    import sys
    sys.exit(main(argparser.parse_args(sys.argv[1:])))
    