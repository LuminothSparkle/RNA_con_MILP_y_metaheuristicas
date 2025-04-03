import sys
import os.path as path
import random

save_path = path.normpath(sys.argv[1])
arch_path = path.join(save_path, 'test_hidden_arch.data')
database_path = path.join(save_path, 'test_database.data')
labels_path = path.join(save_path, 'test_labels.data')
L = random.randint(3,10)
T = random.randint(10,50)
C_0 = random.randint(3,10)
C_L = random.randint(3,10)
with open(arch_path, mode = 'wt', encoding = 'utf-8') as fo :
    sys.stdout = fo
    print(*[random.randint(3,10) for k in range(L - 1)])
    print(*['ReLU' for k in range(L)])
    sys.stdout = sys.__stdout__
with open(database_path, mode = 'wt', encoding = 'utf-8') as fo :
    sys.stdout = fo    
    for t in range(T) :
        print(*[random.uniform(0,10) for i in range(C_0)])
    sys.stdout = sys.__stdout__
with open(labels_path, mode = 'wt', encoding = 'utf-8') as fo :
    sys.stdout = fo    
    for t in range(T) :
        print(*[random.randint(0,1) for i in range(C_L)])
    sys.stdout = sys.__stdout__