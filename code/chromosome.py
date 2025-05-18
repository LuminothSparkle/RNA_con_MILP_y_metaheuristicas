from torch import Tensor
import torch
    
def uniform_crossover(chromosome_a : Tensor, chromosome_b : Tensor, p : float = 0.5) :
    mask = torch.bernoulli(torch.rand_like(chromosome_a), p = p).bool()
    return chromosome_a.bool().bitwise_and(mask).bitwise_or(chromosome_b.bool().bitwise_and(mask.bitwise_not()))
    
def kpoint_crossover(chromosome_a : Tensor, chromosome_b : Tensor, k : int = 1) :
    mask = (torch.arange(chromosome_a.numel()).t() < torch.randperm(chromosome_a.numel())[:k].unsqueeze(dim = 0)).sum(dim = 0).remainder(2).bool()
    return chromosome_a.bool().bitwise_and(mask).bitwise_or(chromosome_b.bool().bitwise_and(mask.bitwise_not()))

def mutation(chromosome : Tensor, p : float = 0.01) :
    return chromosome.bool().bitwise_xor(torch.bernoulli(torch.rand_like(chromosome), p = p).bool())

def get_masks(chromosome : Tensor, weights : list[Tensor]) :
    return [ mask.view_as(weight)  for mask,weight in zip(chromosome.bool().split( [weight.numel() for weight in weights] ), weights) ]
    
