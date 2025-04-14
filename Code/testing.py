import timeit
import torch

a = timeit.repeat('for i in range(100) : a[i]','import torch; a = torch.tensor([[0 for _ in range(100)] for _ in range(100)]);',number=100)
b = timeit.repeat('for i in range(100) : a[i]','import torch; a = torch.zeros(100,100);',number=100)
print(a)
print(b)