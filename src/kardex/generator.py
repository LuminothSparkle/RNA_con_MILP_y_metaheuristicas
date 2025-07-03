"""
Modulo que contiene los metodos para generar la base datos del kardex
para la red neuronal
"""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
from argparse import ArgumentParser
import numpy
import pandas
from pandas import DataFrame

calificacion_dict = {
    "RE": 8,
    "AO": 8,
    "EE": 8,
    "EQ": 8,
    "MB": 10,
    "B": 8,
    "S": 6,
    "NA": 4
}
ORDER = 'IPO'
LEN_ORDER = len(ORDER)


def tri_2_int(code: str) -> int:
    """
    Realiza la conversion del trimestre al entero
    """
    return int(code[0:2]) * LEN_ORDER + ORDER.index(code[2])


def int_2_tri(num: int) -> str:
    """
    Realiza la conversion entera al texto del trimestre
    """
    return f'{num // LEN_ORDER:02d}{ORDER[num % LEN_ORDER]}'


def write_db(
    data_frames: dict[str, DataFrame],
    dir_path: Path, exists_ok: bool = True
):
    """
    Genera los archivos correspondientes para cada carrera usando el
    dataframe, además de comprobar si existe el archivo
    """
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for name, data_frame in data_frames.items():
            file_path = dir_path / f'{name}.csv'
            assert exists_ok or file_path.exists(), (
                f"El archivo {file_path} ya existe"
            )
            futures += [executor.submit(
                data_frame.to_csv,
                file_path,
                index_label='Alumno'
            )]
        for future in futures:
            future.result()


def load_json(file_path: Path):
    """
    A
    """
    with file_path.open(mode='rt', encoding='utf-8') as fp:
        data_dict = json.load(fp)
    return data_dict


def process_kardex(kardex_path: Path, plans_paths: list[Path]):
    """
    Procesa el archivo json y genera un diccionario que contiene
    la información para generar la base de datos del kardex para
    la red neuronal
    """
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures_map = executor.map(load_json, plans_paths)
        kardex_future = executor.submit(load_json, kardex_path)
        career_data = {}
        for plan_path, result in zip(plans_paths, futures_map):
            code = plan_path.stem
            career_data[code] = result
        kardex_dict = kardex_future.result()
    career_dict = {}
    for name, students in kardex_dict.items():
        career_data = {}
        _, code, = name.strip().split()
        if code in career_data:
            dataframe_list = []
            for student in students:
                student_dict = {
                    'estado': '',
                    'trimestre': 0,
                    'creditos obligatorios': 0,
                    'creditos optativos': 0,
                    'promedio general': 0.0
                }
                trimesters = [
                    tri_2_int(intent['trimestre'])
                    for intents in student['ueas'].values()
                    for intent in intents
                ]
                min_tri, max_tri = numpy.min(trimesters), numpy.max(trimesters)
                innactivity_tri = tri_2_int('25I') - max_tri + 1
                if innactivity_tri > 6 or student['estado'] != 'activo':
                    student_dict['trimestre'] = max_tri - min_tri + 1
                    student_dict['estado'] = (
                            student['estado']
                            if student['estado'] == 'egreso'
                            or student['estado'] == 'baja'
                            else 'baja'
                        )
                    student_dict['promedio general'] = numpy.mean([
                        numpy.max([
                            calificacion_dict[intent['calificacion']]
                            for intent in student['ueas'][code]
                        ]) for code in career_data['ueas']
                        if code in student['ueas']
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
                    for code in career_data['ueas']:
                        if code in student['ueas']:
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
                            student_dict[f'{code}_laseva'] = student['ueas'][
                                code
                            ][
                                numpy.argmax([
                                    calificacion_dict[intent['calificacion']]
                                    for intent in student['ueas'][code]
                                ])
                            ]
                            student_dict[f'{code}_inteva'] = numpy.mean([
                                int(intent['evaluacion'] == 'GLO.') + 1
                                for intent in student['ueas'][code]
                            ]).item()
                        else:
                            for suffix in [
                                'initri', 'lastri', 'karcal',
                                'intcal', 'evauea'
                            ]:
                                student_dict[f'{code}_{suffix}'] = 0
                    dataframe_list += [DataFrame(student_dict, index=[0])]
            career_dict[name] = pandas.concat(
                dataframe_list, axis='index',
                join='outer', ignore_index=True
            ).fillna(0)
    return career_dict


def main(args: argparse.Namespace):
    """
    Funcion main para generar la base de datos en archivo csv del archivo
    json del kardex anonimizado
    """
    plans_path = args.load_path / 'Plans'
    kardex_path = args.load_path / 'kardex.json'
    plans_paths = sorted(plans_path.glob('*.json'))
    career_dict = process_kardex(kardex_path, plans_paths)
    args.save_path.mkdir(parents=True, exist_ok=True)
    write_db(career_dict, args.save_path, not args.no_overwrite)


if __name__ == '__main__':
    import sys
    argparser = ArgumentParser()
    argparser.add_argument(
        '--save_path', '-sp',
        type=Path,
        default=Path.cwd() / 'Data' / 'kardex'
    )
    argparser.add_argument(
        '--load_path', '-lp',
        type=Path,
        default=Path.cwd() / 'Data' / 'kardex'
    )
    argparser.add_argument('--no_overwrite', '-no', action='store_true')
    sys.exit(main(argparser.parse_args(sys.argv[1:])))
