"""
A
"""
import numpy
from pathlib import Path
import utility.nn.torchdefault as torchdefault
from utility.io.crossvalidation import load_trainer, load_dataset
from utility.metaheuristics.genetic import (
    genetic_loop, uniform_crossover, mutation, parent_roulette_wheel_selection,
    parent_rank_selection, parent_elitist_selection, poblation_roulette_wheel_selection,
    poblation_elitist_selection, poblation_rank_selection, get_chromosome
)
from utility.metaheuristics.fitness import (
    MaskFitnessCalculator
)

trainer = load_trainer(Path(
    "C:/Users/luisb/Documents/GitHub/RNA_con_MILP_y_metaheuristicas"
    "/Data/Breast Cancer Winsconsin (Diagnostic)/results/datasets/test_pt/"
    "bcw_best.pt"
))
dataset = load_dataset(Path(
    "C:/Users/luisb/Documents/GitHub/RNA_con_MILP_y_metaheuristicas"
    "/Data/Breast Cancer Winsconsin (Diagnostic)/results/datasets/test_pt/"
    "bcw_best.pt"
))

model = trainer.model
for X,Y in torchdefault.sequential_dataloader(
    torchdefault.tensor_dataset(
        dataset.generate_tensors(
            dataset.train_dataframe,
            augment_tensor=True
        )
    )
):
    print(model.inference(X))
    print(Y)
    print(model.loss(X, Y))
print(dataset)
print(model)

base_chromosome = get_chromosome(model.get_weights(), 0.1)
fc = MaskFitnessCalculator(archs=model, trainer=trainer, dataset=dataset)
ss = numpy.random.SeedSequence(0)
generator = numpy.random.default_rng(ss)
torch_generator = torchdefault.generator()
crossover_fn = lambda a,b : uniform_crossover(a, b, 0.5, torch_generator)
mutation_fn = lambda a : mutation(a, 0.01, torch_generator)
aptitude_fn = lambda a : fc.evaluate(a)
par_sel_fn = lambda fit : parent_rank_selection(fit, 50, 0.1, True, generator)
sel_fn = lambda fit1,fit2 : poblation_rank_selection(fit1, fit2, 100, 0.1, True, generator)
genetic_loop(
    base_chromosome, 100, 100, crossover_fn, mutation_fn,
    aptitude_fn, par_sel_fn, sel_fn, torch_generator
)