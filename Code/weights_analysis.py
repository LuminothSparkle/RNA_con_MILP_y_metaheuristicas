import os.path as path
import math
import sys

weights_path = path.normpath(sys.argv[1])
save_path = path.normpath(sys.argv[2])
precision_file_path = path.join(save_path, 'precision.data')
bits_file_path = path.join(save_path, 'bits.data')
w = []
with open(weights_path, mode = 'rt', encoding = 'utf-8') as fo :
    w_M = []
    for line in fo :
        if line.strip().startswith('___') :
            w += [w_M.copy()]
            w_M = []
        else :
            w_M += [[float(string) for string in line.strip().split()]]
    w += [w_M.copy()]
with open(precision_file_path, mode = 'wt', encoding = 'utf-8') as fo :
    sys.stdout = fo
    for layer in w :
        for row in layer :
            print(*[int(math.log2(weight)) for weight in row])
    sys.stdout = sys.__stdout__
with open(bits_file_path, mode = 'wt', encoding = 'utf-8') as fo :
    sys.stdout = fo
    for layer in w :
        for row in layer :
            print(*[4 for weight in row])
    sys.stdout = sys.__stdout__
