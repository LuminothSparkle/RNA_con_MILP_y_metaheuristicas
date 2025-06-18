"""
Modulo que contiene los metodos para generar la base datos del kardex para la red neuronal
"""

import json
from pathlib import Path

import argparse
from argparse import ArgumentParser

import numpy

import pandas
from pandas import DataFrame

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
ORDER = 'IPO'
LEN_ORDER = len(ORDER)

def tri_2_int(code : str)  -> int:
    """
    Realiza la conversion del trimestre al entero
    """
    return int(code[0:2]) * LEN_ORDER + ORDER.index(code[2])

def int_2_tri(num : int)  -> str:
    """
    Realiza la conversion entera al texto del trimestre
    """
    return f'{num // LEN_ORDER:02d}{ORDER[num % LEN_ORDER]}'

def write_db(data_frames : dict[str,DataFrame], dir_path : Path, exists_ok : bool = True) :
    """
    Genera los archivos correspondientes para cada carrera usando el dataframe, además de comprobar
    si existe el archivo
    """
    file_paths = [ dir_path / f'{name}.csv' for name in data_frames ]
    assert exists_ok or all(
        not file_path.exists() for file_path in file_paths
    ), 'Alguno de los archivos ya existe'
    for file_path, data_frame in zip(file_paths, data_frames.values()) :
        data_frame.to_csv(file_path, index_label = 'Alumno')

def process_kardex(kardex_path : Path, plans_paths : list[Path]) :
    """
    Procesa el archivo json y genera un diccionario que con tiene la información para generar
    la base de datos del kardex para la red neuronal
    """
    career_data = {}
    for plan_path in plans_paths :
        code = plan_path.stem
        with plan_path.open(mode = 'rt', encoding = 'utf-8') as fp :
            career_data[code] = json.load(fp)
    with kardex_path.open(mode = 'rt', encoding = 'utf-8') as fp :
        kardex_dict = json.load(fp)
    career_dict = {}
    for name, students in kardex_dict.items() :
        career_data = {}
        _, code, = name.strip().split()
        if code in career_data :
            dataframe_list = []
            for student in students :
                student_dict = {
                    'estado'                : '',
                    'trimestre'             : 0,
                    'creditos obligatorios' : 0,
                    'creditos optativos'    : 0,
                    'promedio general'      : 0.0
                }
                trimesters = [
                    tri_2_int(intent['trimestre'])
                        for intents in student['ueas'].values()
                        for intent in intents
                ]
                min_tri, max_tri = numpy.min(trimesters), numpy.max(trimesters)
                innactivity_tri = tri_2_int('25I') - max_tri + 1
                if innactivity_tri > 6 or student['estado'] != 'activo' :
                    student_dict['trimestre'] = max_tri - min_tri + 1
                    student_dict['estado'] = ( student['estado']
                        if student['estado']  == 'egreso' or student['estado']  == 'baja'
                        else 'baja'
                    )
                    student_dict['promedio general'] = numpy.mean([
                        numpy.max([
                            calificacion_dict[intent['calificacion']]
                                for intent in student['ueas'][code]
                        ]) for code in career_data['ueas'] if code in student['ueas']
                    ]).item()
                    student_dict['creditos obligatorios'] = numpy.sum([
                        career_data['ueas'][code]['creditos']
                            for code in career_data['ueas']
                            if code in student['ueas']
                            and any(
                                intent['calificacion'] != 'NA'
                                    for intent in student['ueas'][code]
                            )
                            and career_data['ueas'][code]['obligatoria']
                    ])
                    student_dict['creditos optativos'] = numpy.sum([
                        career_data['ueas'][code]['creditos']
                            for code in career_data['ueas']
                            if code in student['ueas']
                            and any(
                                intent['calificacion'] != 'NA'
                                    for intent in student['ueas'][code]
                            )
                            and not career_data['ueas'][code]['obligatoria']
                    ])
                    for code in career_data['ueas'] :
                        if code in student['ueas'] :
                            student_dict[f'{code}_initri'] = numpy.min([
                                tri_2_int(intent['trimestre']) - min_tri + 1
                                    for intent in student['ueas'][code]
                            ])
                            student_dict[f'{code}_lastri'] = numpy.max([
                                tri_2_int(intent['trimestre']) - min_tri + 1
                                    for intent in student['ueas'][code]
                            ])
                            student_dict[f'{code}_intcal'] = numpy.mean([
                                calificacion_dict[intent['calificacion']]
                                    for intent in student['ueas'][code]
                            ]).item()
                            student_dict[f'{code}_karcal'] = numpy.max([
                                calificacion_dict[intent['calificacion']]
                                    for intent in student['ueas'][code]
                            ])
                            student_dict[f'{code}_laseva'] = student['ueas'][code][
                                numpy.argmax([
                                    calificacion_dict[intent['calificacion']]
                                        for intent in student['ueas'][code]
                                ])
                            ]
                            student_dict[f'{code}_inteva'] = numpy.mean([
                                int(intent['evaluacion'] == 'GLO.') + 1
                                    for intent in student['ueas'][code]
                            ]).item()
                        else :
                            for suffix in ['initri','lastri','karcal','intcal','evauea'] :
                                student_dict[f'{code}_{suffix}'] = 0
                    dataframe_list += [DataFrame(student_dict, index = [0])]
            career_dict[name] = pandas.concat(
                dataframe_list, axis = 'index',
                join = 'outer', ignore_index = True
            ).fillna(0)
    return career_dict

def main(args : argparse.Namespace) :
    """
    Funcion main para generar la base de datos en archivo csv del archivo
    json del kardex anonimizado
    """
    load_path = Path(args.load_path)
    save_path = Path(args.save_path)
    plans_path = load_path / 'Plans'
    kardex_path = load_path / 'kardex.json'
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
    if not kardex_path.exists() :
        print(f'{kardex_path} doesn\'t exists')
        return None
    if not (plans_path).exists() :
        print(f'{plans_path} doesn\'t exists')
        return None
    elif (plans_path).is_dir() :
        print(f'{plans_path} isn\'t a directory')
        return None
    plans_paths = sorted(plans_path.glob('*.json'))
    career_dict = process_kardex(kardex_path, plans_paths)
    save_path.mkdir(parents = True, exist_ok = True)
    write_db(career_dict, save_path, not args.no_overwrite)

if __name__ == '__main__' :
    import sys
    argparser = ArgumentParser()
    argparser.add_argument('--save_path', '-sp', default = Path.cwd() / 'Data' / 'kardex')
    argparser.add_argument('--load_path', '-lp', default = Path.cwd() / 'Data' / 'kardex')
    argparser.add_argument('--no_overwrite', '-no', action = 'store_true')
    sys.exit(main(argparser.parse_args(sys.argv[1:])))
