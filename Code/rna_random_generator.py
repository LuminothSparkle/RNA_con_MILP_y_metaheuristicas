import sys
from os import path
import torch
import NN_framework as NNF
import pandas_utility as PU

def main(argv) :
    save_path = path.normpath(argv[1])
    arch_path = path.join(save_path, 'random_hidden_arch.data')
    cases_path = path.join(save_path, 'random_cases.data')
    labels_path = path.join(save_path, 'random_labels.data')
    bias_path = path.join(save_path, 'random_bias.data')
    precision_path = path.join(save_path, 'random_precision.data')
    digits_path = path.join(save_path, 'random_digits.data')
    L = torch.randint(3,10,[])
    max_C =  torch.randint(3,10,[])
    T = torch.randint(10,50,[])
    C_0 = torch.randint(3,10,[])
    C_L = torch.randint(3,10,[])
    cases = torch.rand(T,C_0)
    labels = torch.rand(T,C_0)
    hidden_C, phi = NNF.randomHiddenArch(L,max_C)
    C = [ C_0, *hidden_C, C_L ]
    bias = torch.rand(L)
    precision = []
    digits = []
    for k in range(L) :
        precision += [ torch.randint(4,8,[C[k] + 1,C[k + 1]]) ]
        digits += [ torch.randint(4,8,[C[k] + 1,C[k + 1]]) ]
    PU.save_HA(hidden_C,phi,save_path=arch_path)
    PU.save_TDB(tensor=cases,save_path=cases_path)
    PU.save_TDB(tensor=labels,save_path=labels_path)
    PU.save_TDB(bias,save_path=bias_path)
    PU.save_IDB(precision,save_path=precision_path)
    PU.save_IDB(digits,save_path=digits_path)
    

if __name__ == '__main__' :
    main(sys.argv)
