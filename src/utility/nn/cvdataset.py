"""
A
"""
from typing import Any
from abc import abstractmethod
from collections.abc import Iterable, Callable, Generator
from time import perf_counter_ns
import numpy
from numpy import array, ndarray
from pandas import DataFrame, Index
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import BaseCrossValidator
from src.utility.nn.lineal import LinealNN


class CrossvalidationDataset(Dataset):
    """
    Clase abstracta que hereda de dataset de pytorch y debe implementar
    metodos para poder realizar validacion cruzada con scikit learn
    """
    labels: dict[str, Any]
    class_targets: DataFrame
    regression_targets: DataFrame
    features: DataFrame

    @classmethod
    def from_dataframes(
        cls, labels: dict[str, Index],
        train: DataFrame, test: DataFrame, data_augment: int = 0
    ):
        """
        A
        """
    @abstractmethod
    def set_indexes(self, indexes: Iterable[int], is_test: bool = False):
        """
        A
        """
    @abstractmethod
    def split(self, crossvalidator: BaseCrossValidator) -> Generator:
        """
        Metodo abstracto que realiza la separacion de
        acuerdo a un validador cruzado
        """
    @abstractmethod
    def label_decode(self, label: str, pred: ndarray) -> ndarray:
        """
        Metodo que traduce un dataframe a tensor
        """
    @abstractmethod
    def encode(self, target: DataFrame) -> Tensor:
        """
        Metodo que traduce un dataframe a tensor
        """
    @abstractmethod
    def decode(self, pred: Tensor) -> DataFrame:
        """
        Metodo que traduce un tensor a dataframe
        """
    @abstractmethod
    def prediction(
        self, model: Module,
        dataloader: DataLoader | None = None
    ) -> dict:
        """
        Metodo que devuelve las metricas del dataset o dataloader designado
        """
    @abstractmethod
    def loss_fn(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Metodo que implementa la funcion de perdida para el dataset
        """
    @abstractmethod
    def to_dataframe(
        self, subset: str | Iterable[int] | None = None,
        label_type: str | Iterable[int] | None = None, raw: bool = False
    ):
        """
        A
        """
    @abstractmethod
    def data_augment(self, data_augment: int = 0):
        """
        A
        """


def crossvalidate(
    dataset: CrossvalidationDataset, optimizer: tuple[type[Optimizer], dict],
    loss_fn: Callable[[Tensor, Tensor], Tensor], arch: Iterable[int],
    epochs: int, iterations: int, crossvalidator: BaseCrossValidator,
    train_batches: int = 10,
    extra_params: dict[str, Any] | None = None
):
    """
    Realiza la validacion cruzada sobre un conjunto, y un validador
    cruzado de scikit learn utilizando los hyperparametros que
    definen la arquitectura de la red neuronal
    """
    if extra_params is None:
        extra_params = {}
    base_optimizer, optimizer_kwargs = optimizer
    verbose = extra_params['verbose'] if 'verbose' in extra_params else False
    label_list = [
        'loss', 'train time', 'train loss', 'dataset',
        'scheduler', 'model', 'optimizer', 'test index',
        'train index', 'train dataloader', 'test dataloader',
        'parameters'
    ]
    results = {label: [] for label in label_list}
    try:
        for it, (train_dataset, test_dataset) in zip(
            range(iterations),
            dataset.split(crossvalidator=crossvalidator)
        ):
            new_extra_params = extra_params.copy()
            if verbose:
                print(f'iteracion {it}')
            test_size = len(test_dataset)
            train_size = len(train_dataset)
            batch_size = train_size // train_batches
            if torch.cuda.is_available():
                generator = torch.Generator(device='cuda')
            else:
                generator = torch.Generator(device='cpu')
            train_dataloader = DataLoader(
                dataset=train_dataset, shuffle=True,
                batch_size=batch_size,
                drop_last=train_size % train_batches == 1,
                generator=generator
            )
            test_dataloader = DataLoader(
                dataset=test_dataset, batch_size=test_size,
                generator=generator
            )
            new_extra_params['test dataloader'] = test_dataloader
            model = LinealNN(C=arch, hyperparams=extra_params)
            ins_optimizer = base_optimizer(
                model.parameters(), **optimizer_kwargs)
            if 'scheduler' in extra_params:
                base_scheduler, scheduler_kwargs = extra_params['scheduler']
                new_extra_params['scheduler'] = base_scheduler(
                    ins_optimizer, **scheduler_kwargs)
            ns_i = perf_counter_ns()
            train_loss = model.train_loop(
                dataloader=train_dataloader, epochs=epochs,
                optimizer=ins_optimizer, loss_fn=loss_fn,
                extra_params=new_extra_params
            )
            ns_t = perf_counter_ns()
            model.load_state_dict(train_loss['model dict'])
            results['dataset'] += [dataset]
            results['optimizer'] += [ins_optimizer]
            if 'scheduler' in new_extra_params:
                results['scheduler'] += [new_extra_params['scheduler']]
            results['train loss'] += [train_loss]
            results['model'] += [model]
            results['parameters'] += [model.get_total_parameters()]
            results['train time'] += [ns_t - ns_i]
            results['loss'] += [train_loss['best']]
            results['test dataloader'] += [test_dataloader]
            results['train dataloader'] += [train_dataloader]
            results['train index'] += [list(train_dataset.indices)]
            results['test index'] += [list(test_dataset.indices)]
    except KeyboardInterrupt as kbi:
        print(f'{kbi}')
    array_sorted = numpy.argsort(array(results['loss']))
    for label in label_list:
        results[label] = [results[label][it] for it in array_sorted]
    return results
