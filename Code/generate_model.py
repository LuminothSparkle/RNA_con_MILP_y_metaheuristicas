import os.path as path
import sys

# python generate_model.py hidden.data database.data labels.data Coso\Directorio modelo

hidden_arch_path = path.normpath(sys.argv[1])
tensor_path = path.normpath(sys.argv[2])
label_path = path.normpath(sys.argv[3])
save_path = path.normpath(sys.argv[4])
name = sys.argv[5]
arch_path = path.join(save_path, name + '_arch.data')
gurobi_model_path = path.join(save_path, name + '_gm.data')

with open(hidden_arch_path, mode = 'rt', encoding = 'utf-8') as openfileobject:
    C = [int(string) for string in openfileobject.readline().strip().split()]
    phi = [string for string in openfileobject.readline().strip().split()]
L = len(C) + 1
with open(label_path, mode = 'rt', encoding = 'utf-8') as openfileobject:
    lines = openfileobject.readlines()
    T = len(lines)
    C.append(len(lines[0].strip().split()))
with open(tensor_path, mode = 'rt', encoding = 'utf-8') as openfileobject:
    C.insert(0,len(openfileobject.readline().strip().split()))
with open(arch_path, mode = 'wt', encoding = 'utf-8') as openfileobject :
    sys.stdout = openfileobject
    print(L)
    print(*C)
    print(*phi)
    sys.stdout = sys.__stdout__
with open(gurobi_model_path, mode = 'wt', encoding = 'utf-8') as openfileobject :
    sys.stdout = openfileobject
    print(T,L)
    print(*C)
    print(*phi)
    sys.stdout = sys.__stdout__