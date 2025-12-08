from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from time import sleep
from numpy.random import SeedSequence
from sklearn.model_selection import BaseCrossValidator
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import utility.nn.torchdefault as torchdefault
from utility.nn.trainer import TrainerNN
from utility.nn.lineal import LinealNN
from utility.nn.dataset import CsvDataset


class CrossvalidationNN:
    seed: int | list[int]
    trainers: list[TrainerNN]
    trainer_seeds: list[int]
    base_model : LinealNN
    optimizer: type[Optimizer] | None
    scheduler: type[LRScheduler] | None
    optimizer_kwargs: dict | None
    scheduler_kwargs: dict | None
    iterations: int
    workers: int
    tensors: list[dict[str,Tensor]]
    epochs: int
    batch_size: int
    dataset: CsvDataset
    interrupted: bool

    def state_dict(self):
        state_dict = {
            label: self.__dict__[label]
            for label in self.__dict__
            if label not in ['trainers', 'base_model', 'dataset']
        }
        state_dict['trainers']   = [trainer.state_dict() for trainer in self.trainers]
        state_dict['base_model'] = self.base_model.state_dict(keep_vars=True)
        state_dict['dataset']    = self.dataset.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: dict):
        new_state_dict = deepcopy(state_dict)
        trainers = [ TrainerNN() for _ in range(len(new_state_dict['trainers']))]
        for trainer, trainer_state_dict in zip(trainers, new_state_dict['trainers']):
            trainer.load_state_dict(trainer_state_dict)
        base_model = LinealNN()
        base_model.load_state_dict(new_state_dict['base_model'])
        new_state_dict['trainers'] = trainers
        new_state_dict['base_model'] = base_model
        new_state_dict['dataset'] = CsvDataset.from_state_dict(
            state_dict=state_dict['dataset']
        )
        self.__dict__ = new_state_dict

    @classmethod
    def from_dataset(
        cls, dataset: CsvDataset, crossvalidator: BaseCrossValidator,
        base_model: LinealNN,     iterations: int = 10,
        epochs: int = 10,         batch_size: int = 10,
        optimizer: type[Optimizer] | None = None,
        scheculer: type[LRScheduler] | None = None,
        optimizer_kwargs: dict | None = None,
        scheduler_kwargs: dict | None = None,
        connection_drouput = None,
        l1_activation =  None,
        l2_activation =  None,
        l1_weight =  None,
        l2_weight =  None,
        workers: int | None = None, seed: int | None = None
    ):
        cvnn = cls()
        ss = SeedSequence(seed)
        cvnn.dataset = dataset
        cvnn.seed = ss.entropy # type: ignore
        cvnn.optimizer = optimizer
        cvnn.scheduler = scheculer
        cvnn.optimizer_kwargs = optimizer_kwargs
        cvnn.scheduler_kwargs = scheduler_kwargs
        if workers is None:
            workers = 10
        cvnn.base_model = base_model
        cvnn.workers = workers
        cvnn.epochs = epochs
        cvnn.batch_size = batch_size
        tensors = [*dataset.split(crossvalidator=crossvalidator)]
        iterations = min(iterations, len(tensors))
        cvnn.iterations = iterations
        cvnn.trainer_seeds = [seq.entropy for seq in ss.spawn(iterations)] # type: ignore
        cvnn.tensors = [{} for _ in range(iterations)]
        cvnn.trainers = []
        for tensors_data, seed, (train_ftr, train_tgt, test_ftr, test_tgt) in zip(
            cvnn.tensors, cvnn.trainer_seeds, tensors
        ):
            tensors_data['train_features'] = torchdefault.tensor(train_ftr)
            tensors_data['train_targets']  = torchdefault.tensor(train_tgt)
            tensors_data['test_features']  = torchdefault.tensor(test_ftr)
            tensors_data['test_targets']   = torchdefault.tensor(test_tgt)
            cvnn.trainers += [TrainerNN.from_model(
                model=cvnn.base_model.copy(),
                dataset=dataset,
                optimizer=cvnn.optimizer,
                scheduler=cvnn.scheduler,
                scheduler_kwargs=cvnn.scheduler_kwargs,
                optimizer_kwargs=cvnn.optimizer_kwargs,
                epochs=cvnn.epochs,
                batch_size=cvnn.batch_size,
                connection_dropout=connection_drouput,
                l1_activation=l1_activation,
                l2_activation=l2_activation,
                l1_weight=l1_weight,
                l2_weight=l2_weight,
                seed=seed
            )]
        return cvnn

    def crossvalidate(self):
        """
        Realiza la validacion cruzada sobre un conjunto, y un validador
        cruzado de scikit learn utilizando los hyperparametros que
        definen la arquitectura de la red neuronal
        """
        torchdefault.set_defaults()
        self.interrupted = False
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures_list = []
            for tensors, trainer in zip(self.tensors, self.trainers):
                if not trainer.done:
                    futures_list += [executor.submit(
                        trainer.train_model,
                        train_dataset=torchdefault.tensor_dataset(
                            tensors['train_features'], tensors['train_targets']
                        ),
                        test_dataset=torchdefault.tensor_dataset(
                            tensors['test_features'], tensors['test_targets']
                        )
                    )]
            while not all(future.done() for future in futures_list):
                sleep(5)
                for trainer in self.trainers:
                    trainer.interrupted = self.interrupted
            for future in futures_list:
                future.result()
        return all(trainer.done for trainer in self.trainers)
