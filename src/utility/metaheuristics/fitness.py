"""
A
"""
from concurrent.futures import ThreadPoolExecutor
import numpy
import torch
import numpy.random as numpyrand
from numpy import ndarray
from numpy.random import SeedSequence, PCG64, Generator
from torch import Tensor
from torch.utils.data import Subset, DataLoader
from utility.nn.lineal import LinealNN
from utility.metaheuristics.nnred import get_capacity
from utility.nn.trainer import TrainerNN
from utility.nn.dataset import CsvDataset
import utility.nn.torchdefault as torchdefault



def loss_fitness(loss: float):
    """
    A
    """
    return 1000 * numpy.exp(-loss).item()


class WeightFitnessCalculator:
    """
    A
    """
    dataset: CsvDataset
    trainer: TrainerNN
    base_model: LinealNN
    best_loss: float | None

    def __init__(
        self, arch: LinealNN | None = None,
        trainer: TrainerNN | None = None,
        dataset: CsvDataset | None = None
    ) -> None:
        if arch is not None:
            self.base_model = arch
        if trainer is not None:
            self.trainer = trainer
        if dataset is not None:
            self.dataset = dataset
        self.best_loss = None

    def get_best_arch(self, weights: list[ndarray | None]):
        """
        A
        """
        arch = self.base_model.copy()
        arch.set_capacity(get_capacity(weights))
        arch.set_weights(weights) # type: ignore
        self.trainer.set_model(arch)
        test_dataset = torchdefault.tensor_dataset(
            *self.dataset.generate_tensors(
                self.dataset.test_dataframe,
                augment_tensor=False
            )
        )
        dataloader = torchdefault.sequential_dataloader(
            dataset=test_dataset
        )
        losses = [
            arch.loss(features=features, target=targets)
            for features, targets in dataloader
        ]
        loss = numpy.mean(losses).item()
        if self.best_loss is None or self.best_loss > loss:
            self.best_loss = loss
            return {
                'loss': loss_fitness(loss),
                'weights': weights,
                'arch': arch
            }
        self.trainer.train_model(
            train_dataset=torchdefault.tensor_dataset(
                *self.dataset.generate_tensors(
                    dataframe=self.dataset.train_dataframe,
                    augment_tensor=True
                )
            ),
            test_dataset=torchdefault.tensor_dataset(
                *self.dataset.generate_tensors(
                    dataframe=self.dataset.test_dataframe,
                    augment_tensor=False
                )
            )
        )
        if loss > self.trainer.states['fit']['loss']:
            weights = arch.get_weights()
            loss = self.trainer.states['fit']['loss']
        if self.best_loss is None or self.best_loss > loss:
            self.best_loss = loss
        return {
            'loss': loss_fitness(loss),
            'weights': weights,
            'arch': arch
        }
        
    def evaluate(self, weights: list[ndarray | None]):
        result = self.get_best_arch(weights=weights)
        return result['loss'], result['weights']


class MaskFitnessCalculator:
    """
    A
    """
    active_archs: list[LinealNN]
    dataset: CsvDataset
    trainer: TrainerNN
    best_loss: float | None

    def __init__(
        self, archs: LinealNN | list[LinealNN] | None = None,
        trainer: TrainerNN | None = None,
        dataset: CsvDataset | None = None
    ) -> None:
        if archs is not None:
            self.set_archs(archs)
        else:
            self.active_archs = []
        if trainer is not None:
            self.trainer = trainer
        if dataset is not None:
            self.dataset = dataset
        self.best_loss = None

    def add_archs(self, archs: LinealNN | list[LinealNN]):
        self.active_archs = [
            *self.active_archs,
            *([archs] if isinstance(archs, LinealNN) else archs)
        ]

    def set_archs(self, archs: LinealNN | list[LinealNN]):
        self.reset_archs()
        self.add_archs(archs)

    def reset_archs(self):
        self.active_archs = []

    def get_best_arch(self, chromosome: Tensor):
        """
        A
        """
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_list = []
            for arch in self.active_archs:
                weights = arch.get_weights()
                masks = [mask.view_as(weight) for mask, weight in zip(
                    chromosome.split_with_sizes([
                        weight.numel()
                        for weight in weights
                    ]),
                    weights
                )]
                future_list += [executor.submit(
                    arch.copy_with_masks,
                    masks
                )]
            new_archs = [
                future_data.result()
                for future_data in future_list
            ]
            
            def test_closure(model: LinealNN):
                test_dataset = torchdefault.tensor_dataset(
                    *self.dataset.generate_tensors(
                        self.dataset.test_dataframe,
                        augment_tensor=False
                    )
                )
                dataloader = torchdefault.sequential_dataloader(
                    dataset=test_dataset
                )
                losses = [
                    model.loss(features=features, target=targets)
                    for features, targets in dataloader
                ]
                return numpy.mean(losses).item()
            
            def trainer_closure(model: LinealNN):
                test_dataset = torchdefault.tensor_dataset(
                    *self.dataset.generate_tensors(
                        self.dataset.test_dataframe,
                        augment_tensor=False
                    )
                )
                train_dataset = torchdefault.tensor_dataset(
                    *self.dataset.generate_tensors(
                        self.dataset.train_dataframe,
                        augment_tensor=True
                    )
                )
                trainer = self.trainer.copy()
                trainer.set_model(model)
                trainer.train_model(
                    train_dataset=train_dataset,
                    test_dataset=test_dataset
                )
                return trainer
            
            future_list = []
            for model in new_archs:
                future_list += [executor.submit(
                    test_closure,
                    model
                )]
            losses = [future.result() for future in future_list]
            min_i = numpy.argmin(losses)
            if self.best_loss is None or self.best_loss > losses[min_i]:
                self.best_loss = losses[min_i]
                return {
                    'arch': new_archs[min_i], 
                    'loss': loss_fitness(losses[min_i])
                }
            future_list = []
            for model in new_archs:
                future_list += [executor.submit(
                    trainer_closure,
                    model
                )]
            trainers = [future.result() for future in future_list]
            losses = [
                trainer.states['fit']['loss']
                for trainer in trainers
            ]
            min_i = numpy.argmin(losses)
            if self.best_loss is None or self.best_loss > losses[min_i]:
                self.best_loss = losses[min_i]
            return {
                'arch': new_archs[min_i], 
                'loss': loss_fitness(losses[min_i])
            }

    def evaluate(self, chromosome: Tensor):
        """
        A
        """
        return self.get_best_arch(chromosome=chromosome)['loss']
