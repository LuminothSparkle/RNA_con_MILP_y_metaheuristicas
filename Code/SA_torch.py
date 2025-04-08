import torch

def simulated_annealing(black_box_fn, x_0 : torch.Tensor, y_0 : torch.Tensor, temperature_fn, init_temperature : torch.Tensor, neighbor_fn, probability_fn , iterations : int, opt_type : str) :
    if not(opt_type == 'MAX' or opt_type == 'MIN') :
        return x_0, y_0
    x = x_0.copy()
    y = y_0.copy()
    x_best = x.copy()
    y_best = y.copy()
    temperature = init_temperature.copy()
    for _ in range(iterations) :
        neighbor_x = neighbor_fn(x)
        neighbor_y = black_box_fn(neighbor_x)
        if opt_type == 'MIN' and neighbor_y < y_best or opt_type == 'MAX' and neighbor_y > y :
            x_best, y_best = neighbor_x, neighbor_y
        if probability_fn(y,neighbor_y,temperature,opt_type) > torch.rand() :
            x, y = neighbor_x, neighbor_y
        temperature = temperature_fn(temperature)
    return x_best, y_best

def defaultTemperature(temperature) :
    return temperature * 0.1

def defaultProbability(a : torch.Tensor,b : torch.Tensor,temperature : torch.Tensor,opt_type : str) :
    return torch.min(1,torch.exp(-(b - a if opt_type == 'MIN' else a - b)/temperature))