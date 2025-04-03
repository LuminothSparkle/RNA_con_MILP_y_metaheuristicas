import json
import sys
import os.path as path

def Trimester2Int(s) :
    match s[2]:
        case "P":
            return int(s[0:1]) * 3
        case "O":
            return int(s[0:1]) * 3 + 1
        case "I":
            return int(s[0:1]) * 3 + 2

kardex_dir_path = path.normpath(sys.argv[1])
save_path = path.normpath(sys.argv[2])
with open(path.join(kardex_dir_path, 'kardex.json'), mode = 'rt', encoding = 'utf-8') as text_data :
    max_trimesters = 100
    raw_data = json.load(text_data)
    student_list = []
    career_list = []
    uea_dict = {}
    default_trimesters = 12
    evaluacion_dict = {
        "GLO." : 1,
        "REC." : 2
        }
    calificacion_dict = {
        "RE" : 2,
        "AO" : 2,
        "EE" : 2,
        "EQ" : 2,
        "MB" : 3,
        "B" : 2,
        "S" : 1,
        "NA" : -1
        }
    estado_dict = {
        "egreso" : 1,
        "activo" : 0,
        "baja" : -1
        }
    for career,students in raw_data.items() :
        career_list += [career]
        for student in students :
            student_list += [{}]
            student_list[-1]["estado"] = estado_dict[student["estado"]]
            student_list[-1]["career"] = len(career_list)
            student_list[-1]["ueas"] = dict()
            for uea,statuses in student["ueas"].items() :
                uea_dict[uea] = 0 
                student_list[-1]["ueas"][uea] = [ [Trimester2Int(status["trimestre"]), evaluacion_dict[status["evaluacion"]], calificacion_dict[status["calificacion"]]] for status in statuses ]
    for student in student_list :
        min_trim = max_trimesters
        max_trim = 0
        for attempts in student["ueas"].values() :
            min_trim = min(attempts[0][0],min_trim)
            max_trim = max(attempts[-1][0],max_trim)
        student["trimestre inicial"] = min_trim
        student["trimestres"] = max_trim - min_trim
        for attempts in student["ueas"].values() :
            for attempt in attempts :
                attempt[0] -= min_trim - 1
    uea_list = []
    for key in uea_dict.keys() :
        uea_list += [key]
        uea_dict[key] = len(uea_list) - 1
for student in student_list :
    (name, code) = (career_list[student["career"] - 1].strip().split())
    with open(path.join(kardex_dir_path, 'Plans', code + '.json'), mode = 'rt', encoding = 'utf-8') as text_data :
        raw_data = json.load(text_data)
        student["ueas validas"] = [uea for uea in raw_data["ueas"].keys()]

with open(path.join(save_path, 'kardex_labels.data'), mode = 'wt', encoding = 'utf-8') as fo :
    sys.stdout = fo
    for student in student_list :
        for trimester in range(max_trimesters) :
            print(1 if student["estado"] == 1 and student["trimestres"] == trimester else 0, end = " ")
        print("")
    sys.stdout = sys.__stdout__


with open(path.join(save_path, 'kardex_database1.data'), mode = 'wt', encoding = 'utf-8') as fo :
    sys.stdout = fo
    for student in student_list :
        print(student["career"],student["trimestre inicial"],student["trimestres"],end = " ")
        for uea in uea_list :
            if uea in student["ueas"].keys() :
                print(len(student["ueas"][uea]),end = " ")
                for attempt in student["ueas"][uea] :
                    print(attempt[0],attempt[1],attempt[2],end = " ")
                for i in range(5 - len(student["ueas"][uea])) :
                    print("0 0 0",end = " ")
            elif uea in student["ueas validas"] :
                print("0",end = " ")
                for i in range(5) :
                    print("0 0 0",end = " ")
            else :
                print("-1",end = " ")
                for i in range(5) :
                    print("-1 -1 -1",end = " ")
        print("")
    sys.stdout = sys.__stdout__

with open(path.join(save_path, 'kardex_database2.data'), mode = 'wt', encoding = 'utf-8') as fo :
    sys.stdout = fo
    for student in student_list :
        print(student["career"],student["trimestre inicial"],student["trimestres"],end = " ")
        for uea in uea_list :
            if uea in student["ueas"].keys() :
                for attempt in student["ueas"][uea] :
                    print(attempt[0],attempt[1],attempt[2],end = " ")
                for i in range(5 - len(student["ueas"][uea])) :
                    print("0 0 0",end = " ")
            elif uea in student["ueas validas"] :
                for i in range(5) :
                    print("0 0 0",end = " ")
            else :
                for i in range(5) :
                    print("-1 -1 -1",end = " ")
                
        print("")
    sys.stdout = sys.__stdout__

with open(path.join(save_path, 'kardex_database3.data'), mode = 'wt', encoding = 'utf-8') as fo :
    sys.stdout = fo
    for student in student_list :
        print(student["career"],student["trimestre inicial"],student["trimestres"],end = " ")
        for uea in uea_list :
            if uea in student["ueas"].keys() :
                print(len(student["ueas"][uea]),student["ueas"][uea][-1][0],student["ueas"][uea][-1][1],student["ueas"][uea][-1][2],end = " ")
            elif uea in student["ueas validas"] :
                print("0 0 0 0",end = " ")
            else :
                print("-1 -1 -1 -1",end = " ")
        print("")
    sys.stdout = sys.__stdout__

with open(path.join(save_path, 'kardex_database4.data'), mode = 'wt', encoding = 'utf-8') as fo :
    sys.stdout = fo
    for student in student_list :
        print(student["career"],student["trimestre inicial"],student["trimestres"],end = " ")
        for uea in uea_list :
            if uea in student["ueas"].keys() :
                print(student["ueas"][uea][-1][0],student["ueas"][uea][-1][1],student["ueas"][uea][-1][2],end = " ")
            elif uea in student["ueas validas"] :
                print("0 0 0",end = " ")
            else :
                print("-1 -1 -1",end = " ")
        print("")
    sys.stdout = sys.__stdout__


