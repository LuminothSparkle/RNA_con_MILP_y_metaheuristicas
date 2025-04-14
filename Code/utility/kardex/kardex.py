import json
from pandas import (DataFrame, read_csv)
from pathlib import Path
import argparse
from argparse import ArgumentParser

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

def Tri2Int(code : str)  -> int:
    return int(code[0:1]) * len(order) + order.index(code[2])

def Int2Tri(num : int)  -> str:
    return f'{num // len(order):02d}{order[num % len(order)]}'

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
            save_dir.mkdir(parents = True,exist_ok = True)
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
    uea_req_list = ['clave','creditos','obligatoria']
    new_info_list = ['egreso','baja', 'ultimo trimestre activo', 'creditos obligatorios', 'creditos optativos']
    new_info_types = ['Int64','Int64', 'Int64', 'Int64', 'Int64']
    kardex_dict, career_dict = {}, {}
    for name, students in kardex_dict.items() :
        career_data = dict()
        code = name.strip().split()[1]
        career_path = plans_dir / f'{code}.json'
        with career_path.open(mode = 'rt', encoding = 'utf-8') as text_data :
            career_data = json.load(text_data)
        uea_frame = DataFrame([ [uea[data_req] for data_req in uea_req_list] for uea in career_data['ueas'].values() ], columns = uea_req_list)
        uea_frame = uea_frame.astype({'clave' : 'string'}, copy = False)
        uea_frame.set_index('clave', inplace = True)
        uea_info_list = [ f'{code}_{suf}' for suf in ['ini', 'cal', 'int', 'REC', 'GLO'] for code in uea_frame.index.to_list() ]
        uea_info_types = [ dtype for dtype in ['Int64','Float64','Int64','Float64','Float64'] for _ in range(len(uea_frame.index.to_list())) ]
        career_frame = DataFrame(index = range(len(students)), columns = [*new_info_list, *uea_info_list], dtype = 'Int64')
        career_frame = career_frame.astype(dtype = dict(zip([*new_info_list, *uea_info_list],[*new_info_types,*uea_info_types])), copy = False)
        career_frame.fillna(0,inplace = True)
        for idx, student in enumerate(students) :
            career_frame.at[idx,'egreso'] = int(student['estado'] == 'egreso')
            career_frame.at[idx,'baja'] = int(student['estado'] == 'baja')
            first_tri = 10000
            for code, intents in student['ueas'].items() :
                for intent in intents :
                    tri = Tri2Int(intent['trimestre'])
                    first_tri = min(first_tri,tri)
                    career_frame.at[idx,'ultimo trimestre activo'] = max(career_frame.at[idx,'ultimo trimestre activo'],tri)
            career_frame.at[idx,'ultimo trimestre activo'] = career_frame.at[idx,'ultimo trimestre activo'] - first_tri
            for code, intents in student['ueas'].items() :
                if code in uea_frame.index.to_list() :
                    career_frame.at[idx,f'{code}_int'] = len(intents)
                    career_frame.at[idx,f'{code}_ini'] = 10000
                    if any(intent['calificacion'] != 'NA' for intent in intents) :
                        if uea_frame.at[code,'obligatoria'] :
                            career_frame.at[idx,'creditos obligatorios'] = career_frame.at[idx,'creditos obligatorios'] + uea_frame.at[code,'creditos']
                        else :
                            career_frame.at[idx,'creditos optativos'] = career_frame.at[idx,'creditos optativos'] + uea_frame.at[code,'creditos']
                    for intent in intents :
                        tri = Tri2Int(intent['trimestre']) - first_tri
                        career_frame.at[idx,f'{code}_ini'] = min(career_frame.at[idx,f'{code}_ini'],tri)
                        career_frame.at[idx,f'{code}_cal'] = career_frame.at[idx,f'{code}_cal'] + float(calificacion_dict[intent['calificacion']])
                        career_frame.at[idx,f'{code}_REC'] = career_frame.at[idx,f'{code}_REC'] + float(int(intent['evaluacion'] == 'REC.'))
                        career_frame.at[idx,f'{code}_GLO'] = career_frame.at[idx,f'{code}_GLO'] + float(int(intent['evaluacion'] == 'GLO.'))
                    career_frame.at[idx,f'{code}_GLO'] /= len(intents)
                    career_frame.at[idx,f'{code}_REC'] /= len(intents)
                    career_frame.at[idx,f'{code}_cal'] /= len(intents)
        career_dict[name] = career_frame
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
        if not (args.load_path / 'Plans').exists() :
            print(f'{args.load_path / "Plans"} doesn\'t exists')
            return None
        career_dict = process_kardex(load_path)
        write_DB(career_dict, save_path, not args.no_overwrite)
    else :
        separate_targets_and_features(load_path, save_path,not args.no_overwrite)

if __name__ == '__main__' :
    import sys
    sys.exit(main(argparser.parse_args(sys.argv)))
    