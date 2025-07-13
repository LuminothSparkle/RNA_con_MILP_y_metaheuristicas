"""
A
"""
from concurrent.futures import ThreadPoolExecutor
from numpy.random import SeedSequence, PCG64, Generator
import numpy.random as numpyrand
import numpy
from numpy import ndarray
import torch
from torch.utils.data import Subset, DataLoader
from torch import Tensor
from src.utility.nn.lineal import LinealNN
from src.utility.metaheuristics.nnred import get_capacity


def loss_fitness(loss: float):
    """
    A
    """
    return 1000 * numpy.exp(-loss).item()


class WeightFitnessCalculator:
    """
    A
    """
    train_dataset: Subset | None
    test_dataset: Subset | None
    epochs: int
    batch_size: int
    best_loss: float | None

    def __init__(self) -> None:
        self.epochs = 0
        self.batch_size = 1
        self.best_loss = None
        self.train_dataset = None
        self.test_dataset = None

    def set_params(
        self, train: Subset, test: Subset,
        epochs: int = 0, batch_size: int = 1
    ):
        """
        A
        """
        self.train_dataset = train
        self.test_dataset = test
        self.batch_size = batch_size
        self.epochs = epochs

    def evaluate(
        self,
        weights: list[ndarray | None],
        seed: int | Generator | None = None
    ):
        """
        A
        """
        assert (
            self.test_dataset is not None
            and self.train_dataset is not None
        ), (
            "Los dataset de prueba y entrenamiento deben estar establecidos"
        )
        if isinstance(seed, Generator):
            seeder = seed
        else:
            ss = SeedSequence()
            if isinstance(seed, int):
                ss = SeedSequence(entropy=seed)
            seeder = numpyrand.default_rng(PCG64(ss))
        generator = torch.Generator(torch.get_default_device())
        generator.manual_seed(
            seed=seeder.integers(0, 0xffff_ffff_ffff_ffff)  # type: ignore
        )
        arch = LinealNN.from_capacity(get_capacity(weights))
        arch.set_weights(weights)  # type: ignore
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            generator=generator
        )
        loss = arch.test_loop(dataloader)
        if self.best_loss is None or self.best_loss > loss:
            self.best_loss = loss
            return loss_fitness(loss), weights
        train_loss, _, _, fitted_state = arch.train_loop(
            epochs=self.epochs,
            train_dataloader=dataloader,
            test_dataloader=DataLoader(
                self.test_dataset,
                batch_size=len(self.test_dataset),
                shuffle=True,
                generator=generator
            ),
            seed=seeder.integers(
                0, 0xffff_ffff_ffff_ffff
            )  # type: ignore
        )
        if (
            self.best_loss is None
            or self.best_loss > train_loss[fitted_state['epoch']]
        ):
            self.best_loss = train_loss[fitted_state['epoch']]
        if loss > train_loss[fitted_state['epoch']]:
            arch.load_state_dict(fitted_state['state_dict'])
            weights = arch.get_weights()
        return loss_fitness(train_loss[fitted_state['epoch']]), weights


class MaskFitnessCalculator:
    """
    A
    """
    active_archs: list[LinealNN]
    train_dataset: Subset | None
    test_dataset: Subset | None
    epochs: int
    batch_size: int
    best_loss: float | None

    def __init__(self, original: LinealNN) -> None:
        self.epochs = 0
        self.batch_size = 1
        self.best_loss = None
        self.active_archs = [original]
        self.train_dataset = None
        self.test_dataset = None

    def set_params(
        self, train: Subset, test: Subset,
        epochs: int = 0, batch_size: int = 1
    ):
        """
        A
        """
        self.train_dataset = train
        self.test_dataset = test
        self.batch_size = batch_size
        self.epochs = epochs

    def get_best_arch(
        self, chromosome: Tensor,
        seed: int | None = None
    ):
        """
        A
        """
        assert self.test_dataset is not None, (
            "Los dataset de prueba y entrenamiento deben estar establecidos"
        )
        num_arches = len(self.active_archs)
        generators = [
            torch.Generator(torch.get_default_device())
            for _ in range(num_arches)
        ]
        ss = SeedSequence()
        if seed is not None:
            ss = SeedSequence(entropy=seed)
        seeders = [
            numpyrand.default_rng(PCG64(ss_gen))
            for ss_gen in ss.spawn(num_arches)
        ]
        for ss_gen, generator in zip(seeders, generators):
            generator.manual_seed(
                seed=ss_gen.integers(0, 0xffff_ffff_ffff_ffff)  # type: ignore
            )
        with ThreadPoolExecutor(max_workers=10) as executor:
            dataloaders = [
                DataLoader(
                    self.test_dataset,
                    batch_size=len(self.test_dataset),
                    generator=generator
                )
                for generator in generators
            ]
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
            future_list = []
            for arch, dataloader in zip(
                new_archs, dataloaders
            ):
                future_list += [executor.submit(
                    arch.test_loop,
                    dataloader
                )]
            loss_list = [
                future_data.result()
                for future_data in future_list
            ]
        min_i = numpy.argmin(loss_list)
        return new_archs[min_i], loss_list[min_i]

    def evaluate(
        self,
        chromosome: Tensor,
        seed: int | None = None
    ):
        """
        A
        """
        assert (
            self.test_dataset is not None
            and self.train_dataset is not None
        ), (
            "Los dataset de prueba y entrenamiento deben estar establecidos"
        )
        num_arches = len(self.active_archs)
        generators = [
            torch.Generator(torch.get_default_device())
            for _ in range(num_arches)
        ]
        ss = SeedSequence()
        if seed is not None:
            ss = SeedSequence(entropy=seed)
        seeders = [
            numpyrand.default_rng(PCG64(ss_gen))
            for ss_gen in ss.spawn(num_arches)
        ]
        for ss_gen, generator in zip(seeders, generators):
            generator.manual_seed(
                seed=ss_gen.integers(0, 0xffff_ffff_ffff_ffff)  # type: ignore
            )
        with ThreadPoolExecutor(max_workers=10) as executor:
            dataloaders = [
                DataLoader(
                    self.train_dataset,
                    batch_size=self.batch_size,
                    generator=generator
                )
                for generator in generators
            ]
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
            future_list = []
            for arch, dataloader in zip(
                new_archs, dataloaders
            ):
                future_list += [executor.submit(
                    arch.test_loop,
                    dataloader
                )]
            loss_list = [
                future_data.result()
                for future_data in future_list
            ]
        min_i = numpy.argmin(loss_list)
        if self.best_loss is None or self.best_loss > loss_list[min_i]:
            self.best_loss = loss_list[min_i]
            return loss_fitness(loss_list[min_i])
        train_loss, _, _, fitted_state = new_archs[min_i].train_loop(
            epochs=self.epochs,
            train_dataloader=dataloaders[min_i],
            test_dataloader=DataLoader(
                self.test_dataset,
                batch_size=len(self.test_dataset),
                shuffle=True,
                generator=generators[min_i]
            ),
            seed=numpyrand.default_rng(PCG64(ss)).integers(
                0, 0xffff_ffff_ffff_ffff
            )
        )
        if (
            self.best_loss is None
            or self.best_loss > train_loss[fitted_state['epoch']]
        ):
            self.best_loss = train_loss[fitted_state['epoch']]
            new_archs[min_i].load_state_dict(fitted_state['state_dict'])
            self.active_archs += [new_archs[min_i]]
        return loss_fitness(train_loss[fitted_state['epoch']])
