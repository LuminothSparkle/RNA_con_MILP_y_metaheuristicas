"""
A
"""

import torch
from torch import Tensor

def uniform_crossover(chromosome_a : Tensor, chromosome_b : Tensor, p : float = 0.5) :
    """
    A
    """
    mask = torch.bernoulli(torch.rand_like(chromosome_a), p = p).bool()
    return chromosome_a.bool().bitwise_and(mask).bitwise_or(
        chromosome_b.bool().bitwise_and(mask.bitwise_not())
    )

def kpoint_crossover(chromosome_a : Tensor, chromosome_b : Tensor, k : int = 1) :
    """
    A
    """
    mask = (
        torch.arange(chromosome_a.numel()).t() <
        torch.randperm(chromosome_a.numel())[:k].unsqueeze(dim = 0)
    ).sum(dim = 0).remainder(2).bool()
    return chromosome_a.bool().bitwise_and(mask).bitwise_or(
        chromosome_b.bool().bitwise_and(mask.bitwise_not())
    )

def mutation(chromosome : Tensor, p : float = 0.01) :
    """
    A
    """
    return chromosome.bool().bitwise_xor(torch.bernoulli(torch.rand_like(chromosome), p = p).bool())

def get_masks(chromosome : Tensor, weights : list[Tensor]) :
    """
    A
    """
    return [ mask.view_as(weight)
        for mask, weight in zip(chromosome.bool().split(
            [weight.numel() for weight in weights]
        ), weights)
    ]

def get_chromosome(masks : list[Tensor]) -> Tensor :
    """
    A
    """
    return torch.concat(tuple(tensor.flatten() for tensor in masks))
