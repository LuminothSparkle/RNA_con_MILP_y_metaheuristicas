import os.path as path
import math
import sys

sol_path = path.normpath(sys.argv[1])
arch_path = path.normpath(sys.argv[2])
save_path = path.normpath(sys.argv[3])
save_file_path = path.join(save_path, 'gurobi_weights.data')
precis = 1
with open(arch_path, mode = 'rt', encoding = 'utf-8') as fo :
    L = int(fo.readline().strip())
    C = [int(string) for string in fo.readline().strip().split()]
    phi = [string for string in fo.readline().strip().split()]
w = [[[float(0) for j in range(C[k + 1])] for i in range(C[k] + 1)] for k in range(L)]
with open(sol_path, mode = 'rt', encoding = 'utf-8') as fo :
    fo.readline()
    for line in fo :
        name, value = line.strip().split()
        if name.startswith('b_') :
            value = float(value)
            k,i,j,l = [ int(string) for string in name.lstrip('b_').split('_') ]
            w[k][i][j] += value * math.exp2(l - precis)
with open(save_file_path, mode = 'wt', encoding = 'utf-8') as fo :
    sys.stdout = fo
    for layer in w :
        for rows in layer :
            print(*rows)
        print('___')
    sys.stdout = sys.__stdout__

