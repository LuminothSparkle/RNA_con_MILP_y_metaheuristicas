from copy import deepcopy
from time import perf_counter_ns
import numpy
import torch
import numpy.random as numpyrand
from numpy import uint64
from numpy.random import SeedSequence
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset
import utility.nn.torchdefault as torchdefault
from utility.nn.lineal import LinealNN


class TrainerNN:
    batch_size: int
    epochs: int
    model: LinealNN
    base_scheduler: type[LRScheduler] | None
    base_optimizer: type[Optimizer]
    scheduler: LRScheduler | None
    optimizer: Optimizer
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

    def state_dict(self):
        state_dict = deepcopy(self.__dict__)
        state_dict['model'] = self.model.state_dict()
        state_dict['optimizer'] = self.optimizer.state_dict()
        if self.scheduler is not None:
            state_dict['scheduler'] = self.scheduler.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: dict):
        new_state_dict = deepcopy(state_dict)
        model = LinealNN()
        model.load_state_dict(new_state_dict['model'])
        new_state_dict['model'] = model
        optimizer = new_state_dict['base_optimizer'](model.parameters(), new_state_dict['optimizer_kwargs'])
        optimizer.load_state_dict(new_state_dict['optimizer'])
        scheduler = new_state_dict['base_scheduler'](optimizer, new_state_dict['scheduler_kwargs'])
        scheduler.load_state_dict(new_state_dict['scheduler'])
        new_state_dict['model'] = model
        new_state_dict['optimizer'] = optimizer
        new_state_dict['scheduler'] = scheduler
        self.__dict__ = new_state_dict

    @classmethod
    def from_model(
        cls, model: LinealNN,
        optimizer: type[Optimizer]   | None = None,
        scheduler: type[LRScheduler] | None = None,
        batch_size: int = 10, epochs: int = 10,
        overfit_tolerance: int | None = None,
        optimizer_kwargs: dict | None = None,
        scheduler_kwargs: dict | None = None,
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
            name: {'state_dict': {}, 'loss': None, 'epoch': None}
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
        self.done = (self.actual_epoch + 1 >= self.epochs)
    
    def train_epoch(self, dataloader: DataLoader, seed: int | None = None):
        """
        A
        """
        if seed is None:
            seed = torch.seed()
        else:
            torch.manual_seed(seed=seed)
        batch_loss = []
        for features, targets in dataloader:
            losses = self.model.train_closure(features, targets)
            batch_loss += [losses['norm_loss']]
            self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return numpy.mean(batch_loss).item()

    def test_epoch(self, dataloader: DataLoader):
        """
        A
        """
        torchdefault.set_defaults()
        with torch.inference_mode():
            loss = numpy.mean([
                self.model.loss_layer(self.model(X), y).cpu().detach().item()
                for X, y in dataloader
            ]).item()
        return loss

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
                self.states['fit']['state_dict'] = deepcopy(self.model.state_dict())
            else:
                self.overfit = True
            if self.states['overfit']['loss'] is None or loss > self.states['overfit']['loss']:
                self.states['overfit']['epoch'] = epoch
                self.states['overfit']['loss'] = loss
                self.states['overfit']['state_dict'] = deepcopy(self.model.state_dict())
            if self.overfit_tolerance is not None:
                overfit_counter -= 1  # type: ignore
                if overfit_counter < 0:
                    break
            self.actual_epoch = epoch
            if self.interrupted:
                break
        self.states['last']['epoch'] = self.actual_epoch
        self.states['last']['loss'] = loss
        self.states['last']['state_dict'] = deepcopy(self.model.state_dict())
        self.model.load_state_dict(
            state_dict=deepcopy(self.states['fit']['state_dict']),
            strict=False,
            assign=True
        )

    def copy(self):
        return deepcopy(self)
