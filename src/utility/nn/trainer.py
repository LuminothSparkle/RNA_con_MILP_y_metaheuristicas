from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from time import perf_counter_ns
import torch
import numpy
import numpy.random as numpyrand
from numpy import uint64
from numpy.random import SeedSequence
from torch import Tensor
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset
import utility.nn.torchdefault as torchdefault
from utility.nn.lineal import LinealNN, process_layer_param
from utility.nn.dataset import CsvDataset


class TrainerNN:
    model: LinealNN
    scheduler: LRScheduler | None
    optimizer: Optimizer
    dataset: CsvDataset
    batch_size: int
    epochs: int
    base_scheduler: type[LRScheduler] | None
    base_optimizer: type[Optimizer]
    optimizer_kwargs: dict
    scheduler_kwargs: dict
    overfit_tolerance: int | None
    actual_epoch: int
    states: dict
    test_loss: list[float]
    train_loss: list[float]
    start_loss: float
    train_time: int
    overfit: bool
    seed: int
    epoch_seeds: list[int]
    done: bool
    interrupted: bool
    l1_activation: list
    l2_activation: list
    l1_weight: list
    l2_weight: list
    connection_dropout: list

    def state_dict(self):
        state_dict = {
            label: self.__dict__[label]
            for label in self.__dict__
            if label not in ['model', 'optimizer', 'scheduler', 'dataset']
        }
        state_dict['model'] = self.model.state_dict(keep_vars=True)
        state_dict['optimizer'] = self.optimizer.state_dict()
        state_dict['dataset'] = self.dataset.state_dict()
        if self.scheduler is not None:
            state_dict['scheduler'] = self.scheduler.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: dict):
        new_state_dict = deepcopy(state_dict)
        model = LinealNN()
        model.load_state_dict(new_state_dict['model'], strict=True, assign=True)
        optimizer = new_state_dict['base_optimizer'](
            model.parameters(), **new_state_dict['optimizer_kwargs']
        )
        scheduler = new_state_dict['base_scheduler'](optimizer, **new_state_dict['scheduler_kwargs'])
        scheduler.load_state_dict(new_state_dict['scheduler'])
        optimizer.load_state_dict(new_state_dict['optimizer'])
        new_state_dict['dataset'] = CsvDataset.from_state_dict(new_state_dict['dataset'])
        new_state_dict['model'] = model
        new_state_dict['optimizer'] = optimizer
        new_state_dict['scheduler'] = scheduler
        self.__dict__ = new_state_dict

    @classmethod
    def from_model(
        cls, model: LinealNN,
        dataset: CsvDataset,
        optimizer: type[Optimizer]   | None = None,
        scheduler: type[LRScheduler] | None = None,
        batch_size: int = 10, epochs: int = 10,
        overfit_tolerance: int | None = None,
        optimizer_kwargs: dict | None = None,
        scheduler_kwargs: dict | None = None,
        l1_activation: dict | list | None = None,
        l2_activation: dict | list | None = None,
        l1_weight: dict | list | None = None,
        l2_weight: dict | list | None = None,
        connection_dropout: dict | list | None = None,
        seed: int | None = None
    ): 
        ss = SeedSequence(seed)
        trainer = cls()
        trainer.seed = ss.entropy # type: ignore
        if optimizer is not None:
            trainer.base_optimizer = optimizer
        else:
            trainer.base_optimizer = Adam
        trainer.base_scheduler = scheduler
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        if scheduler_kwargs is None:
            scheduler_kwargs = {}
        trainer.optimizer_kwargs = optimizer_kwargs
        trainer.scheduler_kwargs = scheduler_kwargs
        trainer.batch_size = batch_size
        trainer.overfit_tolerance = overfit_tolerance
        trainer.epochs = epochs
        trainer.epoch_seeds = [seq.entropy for seq in ss.spawn(trainer.epochs)] # type: ignore
        trainer.set_model(model)
        trainer.dataset = dataset
        trainer.l1_activation = process_layer_param(param=l1_activation, layers=model.layers)
        trainer.l2_activation = process_layer_param(param=l2_activation, layers=model.layers)
        trainer.l1_weight = process_layer_param(param=l1_weight, layers=model.layers)
        trainer.l2_weight = process_layer_param(param=l2_weight, layers=model.layers)
        trainer.connection_dropout = process_layer_param(
            param=connection_dropout, layers=model.layers
        )
        return trainer

    def set_model(self, model: LinealNN):
        self.model = model
        self.optimizer = self.base_optimizer(
            params=self.model.parameters(),
            **self.optimizer_kwargs
        )
        if self.base_scheduler is not None:
            self.scheduler = self.base_scheduler(self.optimizer, **self.scheduler_kwargs)
        else:
            self.scheduler = None
        self.overfit = False
        self.states = {
            name: {'state_dict': None, 'loss': None, 'epoch': None}
            for name in ['fit', 'overfit', 'last']
        }
        self.done = False
        self.actual_epoch = 0
        self.test_loss  = [0.0 for _ in range(self.epochs)]
        self.train_loss = [0.0 for _ in range(self.epochs)]
        self.train_time = 0
        self.start_loss = 0.0

    def train_model(self, train_dataset: Dataset, test_dataset: Dataset):
        torchdefault.set_defaults()
        generator = torchdefault.generator(
            int(numpyrand.default_rng(self.seed).integers(
                low=0, high=0xffff_ffff_ffff_ffff,
                endpoint=True, dtype=uint64
            ))
        )
        train_dataloader = DataLoader(
            dataset=train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            drop_last=True,
            generator=generator
        )
        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=None
        )
        ns_i = perf_counter_ns()
        self.train_loop(
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader
        )
        ns_t = perf_counter_ns()
        self.train_time += ns_t - ns_i
        self.done = self.actual_epoch + 1 >= self.epochs

    def l2_weight_regularization(self):
        """
        A
        """
        torchdefault.set_defaults()
        l2_layers = [
            (
                norm_weight *
                linear_layer.weight.square().mean()  # type: ignore
            ).unsqueeze(dim=0)
            for linear_layer, norm_weight in zip(
                self.model.linear_layers,
                self.l2_weight
            ) if norm_weight is not None
        ]
        if len(l2_layers) > 0:
            return torch.concat(l2_layers).mean()
        return torchdefault.tensor(0)

    def l1_activation_regularization(self):
        """
        A
        """
        torchdefault.set_defaults()
        l1_layers = [
            (
                norm_weight
                * activation.abs().mean()  # type: ignore
            ).unsqueeze(dim=0)
            for activation, norm_weight in zip(
                self.model.activations,
                self.l1_activation
            ) if norm_weight is not None
        ]
        if len(l1_layers) > 0:
            return torch.concat(l1_layers).mean()
        return torchdefault.tensor(0)

    def l2_activation_regularization(self):
        """
        A
        """
        torchdefault.set_defaults()
        l2_layers = [
            (
                norm_weight
                * activation.square().mean()  # type: ignore
            ).unsqueeze(dim=0)
            for activation, norm_weight in zip(
                self.model.activations,
                self.l2_activation
            ) if norm_weight is not None
        ]
        if len(l2_layers) > 0:
            return torch.concat(l2_layers).mean()
        return torchdefault.tensor(0)

    def l1_weight_regularization(self):
        """
        A
        """
        torchdefault.set_defaults()
        l1_layers = [
            (
                norm_weight
                * linear_layer.weight.abs().mean()  # type: ignore
            ).unsqueeze(dim=0)
            for linear_layer, norm_weight in zip(
                self.model.linear_layers,
                self.l1_weight
            ) if norm_weight is not None
        ]
        if len(l1_layers) > 0:
            return torch.concat(l1_layers).mean()
        return torchdefault.tensor(0)

    def inference(self, x: Tensor):
        """
        Ciclo principal para obtener solamente la inferencia de la red neuronal
        sobre un conjunto de datos
        """
        torchdefault.set_defaults()
        with torch.inference_mode():
            return self.dataset.inference_fn(
                self.model(x)
            ).cpu().detach().numpy()

    def loss(self, features: Tensor, target: Tensor):
        torchdefault.set_defaults()
        with torch.inference_mode():
            return self.dataset.loss_fn(
                self.model(features),
                target
            ).cpu().detach().numpy()

    def train_epoch(self, dataloader: DataLoader, seed: int | None = None):
        torchdefault.set_defaults()
        if seed is None:
            seed = torch.seed()
        else:
            torch.manual_seed(seed=seed)
        batch_loss = []
        raw_loss = []
        self.model.train()
        for features, targets in dataloader:
            weights_copy = []
            for linear_layer, connect_dropout, mask in zip(
                self.model.linear_layers,
                self.connection_dropout,
                self.model.masks_layer
            ):
                if (
                    connect_dropout is not None and
                    mask is not None and
                    isinstance(linear_layer, torch.nn.Linear)
                ):
                    weight = linear_layer.weight
                    connection_mask = torch.full_like(
                        weight,  # type: ignore
                        connect_dropout,
                        dtype=torch.get_default_dtype(),
                        device=torch.get_default_device()
                    ).bernoulli().bool()
                    weight_mask = mask.bitwise_and(connection_mask)
                    with torch.no_grad():
                        linear_layer.weight.copy_(
                            weight.clone() * weight_mask.type_as(  # type: ignore
                                weight  # type: ignore
                            )
                        )
                    weights_copy += [weight]
            self.optimizer.zero_grad()
            loss = self.dataset.loss_fn(self.model(features), targets)
            with ThreadPoolExecutor(max_workers=4) as executor:
                l1a_future = executor.submit(
                    self.l1_activation_regularization
                )
                l2a_future = executor.submit(
                    self.l2_activation_regularization
                )
                l1w_future = executor.submit(
                    self.l1_weight_regularization
                )
                l2w_future = executor.submit(
                    self.l2_weight_regularization
                )
                raw_loss += [loss.cpu().detach().item()]
                loss += (
                    l1w_future.result() + l2w_future.result() +
                    l1a_future.result() + l2a_future.result()
                )
            loss.backward()
            copys = iter(weights_copy)
            for linear_layer, connect_dropout, mask in zip(
                self.model.linear_layers,
                self.connection_dropout,
                self.model.masks_layer
            ):
                if (
                    connect_dropout is not None and
                    mask is not None and
                    isinstance(linear_layer, torch.nn.Linear)
                ):
                    with torch.no_grad():
                        linear_layer.weight.copy_(next(copys))  # type: ignore
            for linear_layer, mask in zip(
                self.model.linear_layers, self.model.masks_layer
            ):
                if (
                    isinstance(linear_layer, torch.nn.Linear) and
                    linear_layer.weight.grad is not None
                    and mask is not None
                ):
                    with torch.no_grad():
                        linear_layer.weight.grad.copy_(  # type: ignore
                            linear_layer.weight.grad
                            * mask.type_as(  # type: ignore
                                linear_layer.weight.grad  # type: ignore
                            )
                        )
            self.optimizer.step()
            batch_loss += [loss.cpu().detach().item()]
        if self.scheduler is not None:
            self.scheduler.step()
        self.model.eval()
        return numpy.mean(batch_loss).item()

    def test_epoch(self, dataloader: DataLoader):
        """
        A
        """
        torchdefault.set_defaults()
        with torch.inference_mode():
            return numpy.mean([
                self.dataset.loss_fn(self.model(X), y).cpu().detach().item()
                for X, y in dataloader
            ]).item()

    def train_loop(
        self, train_dataloader: DataLoader,
        test_dataloader: DataLoader | None = None,
    ):
        """
        A
        """
        self.interrupted = False
        if test_dataloader is None:
            test_dataloader = train_dataloader
        overfit_counter = self.overfit_tolerance
        loss = 0.0
        if self.states['last']['state_dict'] is not None:
            self.model.load_state_dict(
                deepcopy(self.states['last']['state_dict']),
                strict=False,
                assign=True
            )
        for epoch, seed in zip(range(self.actual_epoch, self.epochs), self.epoch_seeds):
            self.train_loss[epoch] = self.train_epoch(
                train_dataloader,
                int(numpyrand.default_rng(seed).integers(
                    low=0, high=0xffff_ffff_ffff_ffff,
                    endpoint=True, dtype=uint64
                ))
            )
            print(f'train_loss {epoch}: {self.train_loss[epoch]}')
            loss = self.test_epoch(test_dataloader)
            self.test_loss[epoch] = loss
            print(f'test_loss {epoch}: {self.test_loss[epoch]}')
            if self.states['fit']['loss'] is None or loss < self.states['fit']['loss']:
                self.overfit = False
                overfit_counter = self.overfit_tolerance
                self.states['fit']['epoch'] = epoch
                self.states['fit']['loss'] = loss
                self.states['fit']['state_dict'] = deepcopy(self.model.state_dict(keep_vars=True))
            else:
                self.overfit = True
            if self.states['overfit']['loss'] is None or loss > self.states['overfit']['loss']:
                self.states['overfit']['epoch'] = epoch
                self.states['overfit']['loss'] = loss
                self.states['overfit']['state_dict'] = deepcopy(
                    self.model.state_dict(keep_vars=True)
                )
            if self.overfit_tolerance is not None:
                overfit_counter -= 1  # type: ignore
                if overfit_counter < 0:
                    break
            self.actual_epoch = epoch
            if self.interrupted:
                break
        self.states['last']['epoch'] = self.actual_epoch
        self.states['last']['loss'] = loss
        self.states['last']['state_dict'] = deepcopy(self.model.state_dict(keep_vars=True))
        self.model.load_state_dict(
            state_dict=deepcopy(self.states['fit']['state_dict']),
            strict=False,
            assign=True
        )

    def copy(self):
        return deepcopy(self)
