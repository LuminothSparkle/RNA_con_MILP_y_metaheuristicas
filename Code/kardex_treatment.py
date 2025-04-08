import json
import sys
import os.path as path

class KardexData :
    def Trimester2Int(self,s) :
        match s[2]:
            case "P":
                return int(s[0:1]) * 3
            case "O":
                return int(s[0:1]) * 3 + 1
            case "I":
                return int(s[0:1]) * 3 + 2
        
    def __init__(self,kardex_dir_path) :
        with open(path.join(kardex_dir_path, 'kardex.json'), mode = 'rt', encoding = 'utf-8') as text_data :
            self.max_trimesters = 100
            raw_data = json.load(text_data)
            self.student_list = []
            self.career_list = []
            self.uea_dict = {}
            self.default_trimesters = 12
            self.evaluacion_dict = {
                "GLO." : 1,
                "REC." : 2
                }
            self.calificacion_dict = {
                "RE" : 2,
                "AO" : 2,
                "EE" : 2,
                "EQ" : 2,
                "MB" : 3,
                "B" : 2,
                "S" : 1,
                "NA" : -1
                }
            self.estado_dict = {
                "egreso" : 1,
                "activo" : 0,
                "baja" : -1
                }
            for career,students in raw_data.items() :
                self.career_list += [career]
                for student in students :
                    self.student_list += [{}]
                    self.student_list[-1]["estado"] = self.estado_dict[student["estado"]]
                    self.student_list[-1]["career"] = len(self.career_list)
                    self.student_list[-1]["ueas"] = dict()
                    for uea,statuses in student["ueas"].items() :
                        self.uea_dict[uea] = 0 
                        self.student_list[-1]["ueas"][uea] = [ [self.Trimester2Int(status["trimestre"]), self.evaluacion_dict[status["evaluacion"]], self.calificacion_dict[status["calificacion"]]] for status in statuses ]
            for student in self.student_list :
                min_trim = self.max_trimesters
                max_trim = 0
                for attempts in student["ueas"].values() :
                    min_trim = min(attempts[0][0],min_trim)
                    max_trim = max(attempts[-1][0],max_trim)
                student["trimestre inicial"] = min_trim
                student["trimestres"] = max_trim - min_trim
                for attempts in student["ueas"].values() :
                    for attempt in attempts :
                        attempt[0] -= min_trim - 1
            self.uea_list = []
            for key in self.uea_dict.keys() :
                self.uea_list += [key]
                self.uea_dict[key] = len(self.uea_list) - 1
        for student in self.student_list :
            (_, code) = (self.career_list[student["career"] - 1].strip().split())
            with open(path.join(kardex_dir_path, 'Plans', code + '.json'), mode = 'rt', encoding = 'utf-8') as text_data :
                raw_data = json.load(text_data)
                student["ueas validas"] = [uea for uea in raw_data["ueas"].keys()]        
    
    def generate_labels(self,save_path) :
        with open(path.join(save_path, 'kardex_labels.data'), mode = 'wt', encoding = 'utf-8') as fo :
            sys.stdout = fo
            for student in self.student_list :
                for trimester in range(self.max_trimesters) :
                    print(1 if student["estado"] == 1 and student["trimestres"] == trimester else 0, end = " ")
                print("")
            sys.stdout = sys.__stdout__
        
    def generate_tensor(self,save_path) :
        with open(path.join(save_path, 'kardex_database1.data'), mode = 'wt', encoding = 'utf-8') as fo :
            sys.stdout = fo
            for student in self.student_list :
                print(student["career"],student["trimestre inicial"],student["trimestres"],end = " ")
                for uea in self.uea_list :
                    if uea in student["ueas"].keys() :
                        print(len(student["ueas"][uea]),end = " ")
                        for attempt in student["ueas"][uea] :
                            print(attempt[0],attempt[1],attempt[2],end = " ")
                        for _ in range(5 - len(student["ueas"][uea])) :
                            print("0 0 0",end = " ")
                    elif uea in student["ueas validas"] :
                        print("0",end = " ")
                        for _ in range(5) :
                            print("0 0 0",end = " ")
                    else :
                        print("-1",end = " ")
                        for _ in range(5) :
                            print("-1 -1 -1",end = " ")
                print("")
            sys.stdout = sys.__stdout__

        with open(path.join(save_path, 'kardex_database2.data'), mode = 'wt', encoding = 'utf-8') as fo :
            sys.stdout = fo
            for student in self.student_list :
                print(student["career"],student["trimestre inicial"],student["trimestres"],end = " ")
                for uea in self.uea_list :
                    if uea in student["ueas"].keys() :
                        for attempt in student["ueas"][uea] :
                            print(attempt[0],attempt[1],attempt[2],end = " ")
                        for _ in range(5 - len(student["ueas"][uea])) :
                            print("0 0 0",end = " ")
                    elif uea in student["ueas validas"] :
                        for _ in range(5) :
                            print("0 0 0",end = " ")
                    else :
                        for _ in range(5) :
                            print("-1 -1 -1",end = " ")
                        
                print("")
            sys.stdout = sys.__stdout__

        with open(path.join(save_path, 'kardex_database3.data'), mode = 'wt', encoding = 'utf-8') as fo :
            sys.stdout = fo
            for student in self.student_list :
                print(student["career"],student["trimestre inicial"],student["trimestres"],end = " ")
                for uea in self.uea_list :
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
            for student in self.student_list :
                print(student["career"],student["trimestre inicial"],student["trimestres"],end = " ")
                for uea in self.uea_list :
                    if uea in student["ueas"].keys() :
                        print(student["ueas"][uea][-1][0],student["ueas"][uea][-1][1],student["ueas"][uea][-1][2],end = " ")
                    elif uea in student["ueas validas"] :
                        print("0 0 0",end = " ")
                    else :
                        print("-1 -1 -1",end = " ")
                print("")
            sys.stdout = sys.__stdout__
