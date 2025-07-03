"""
A
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Any
from abc import abstractmethod
from collections.abc import Iterable, Generator
from time import perf_counter_ns
from matplotlib.pylab import PCG64, SeedSequence
import numpy
from numpy import array, ndarray
from pandas import DataFrame, Index
from sklearn.model_selection import BaseCrossValidator
import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset, Subset
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
        train: DataFrame, test: DataFrame,
        crossvalidator: BaseCrossValidator | None = None,
        data_augment: int = 0
    ):
        """
        A
        """
    @abstractmethod
    def set_test_indexes(self, indexes: Iterable[int]):
        """
        A
        """
    @abstractmethod
    def set_train_indexes(self, indexes: Iterable[int]):
        """
        A
        """
    @abstractmethod
    def split(self) -> Generator:
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
        self, model: LinealNN,
        indices: list[int] | None = None
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
    def inference_function(self, pred: Tensor) -> Tensor:
        """
        Metodo que implementa la funcion de perdida para el dataset
        """
    @abstractmethod
    def to_dataframe(
        self, subset: str | Iterable[int] | None = None,
        label_type: str | Iterable[str] | None = None, raw: bool = False
    ):
        """
        A
        """
    @abstractmethod
    def data_augment(self, data_augment: int = 0):
        """
        A
        """
    @abstractmethod
    def generator_dict(self) -> dict:
        """
        A
        """
    @classmethod
    def from_generator_dict(cls, generator: dict):
        """
        A
        """


def crossvalidate(
    dataset: CrossvalidationDataset,
    optimizer: tuple[type[Optimizer], list, dict],
    capacity: list[int],
    epochs: int, iterations: int,
    train_batches: int = 10,
    **kwargs
):
    """
    Realiza la validacion cruzada sobre un conjunto, y un validador
    cruzado de scikit learn utilizando los hyperparametros que
    definen la arquitectura de la red neuronal
    """
    label_list = [
        'models_loss', 'models_train_loss', 'models_test_loss',
        'models_start_loss', 'models_train_time', 'models_fitted_state',
        'models_overfitted_state', 'models', 'models_test_indices',
        'models_train_indices', 'models_parameters', 'dataset_generators',
        'seed'
    ]
    results = {label: [] for label in label_list}
    ss = SeedSequence()
    if 'seed' in kwargs and kwargs['seed'] is not None:
        ss = SeedSequence(entropy=kwargs['seed'])
    max_workers = 10
    if 'threads' in kwargs and kwargs['threads'] is not None:
        max_workers = kwargs['threads']
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures_list = []
        for (train_dataset, test_dataset), ss_gen in zip(
            dataset.split(),
            ss.spawn(iterations)
        ):
            results['dataset_generators'] += [dataset.generator_dict()]
            results['models_train_indices'] += [train_dataset.indices]
            results['models_test_indices'] += [test_dataset.indices]
            results['seed'] += [ss_gen.entropy]

            def model_train_loop(
                train_dataset: Subset,
                test_dataset: Subset,
                ss: SeedSequence
            ):
                torch.set_default_device('cpu')
                torch.set_default_dtype(torch.double)
                if torch.cuda.is_available():
                    torch.set_default_device('cuda')
                    torch.set_default_dtype(torch.double)
                generator = torch.Generator(device=torch.get_default_device())
                gen = numpy.random.default_rng(PCG64(ss))
                generator.manual_seed(
                    gen.integers(
                        0, 0xffff_ffff_ffff_ffff  # type: ignore
                    )
                )
                test_size = len(test_dataset)
                train_size = len(train_dataset)
                batch_size = train_size // train_batches
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
                model = LinealNN.from_capacity(capacity=capacity, **kwargs)
                op_base, op_args, op_kwargs = optimizer
                model.set_optimizer(op_base, *op_args, **op_kwargs)
                if 'scheduler' in kwargs:
                    sc_base, sc_args, sc_kwargs = kwargs['scheduler']
                    model.set_scheduler(sc_base, *sc_args, **sc_kwargs)
                ns_i = perf_counter_ns()
                tup = model.train_loop(
                    epochs=epochs,
                    train_dataloader=train_dataloader,
                    test_dataloader=test_dataloader,
                    seed=gen.integers(
                        0, 0xffff_ffff_ffff_ffff  # type: ignore
                    )
                )
                ns_t = perf_counter_ns()
                train_loss, test_loss, start_loss, fitted_state = tup
                return (
                    train_loss, test_loss, start_loss,
                    fitted_state, ns_t - ns_i, model
                )
            futures_list += [executor.submit(
                model_train_loop,
                train_dataset,
                test_dataset,
                ss_gen
            )]
        for future_data in futures_list:
            (
                train_loss, test_loss, start_loss,
                fitted_state, train_time, model
            ) = future_data.result()
            results['models_train_loss'] += [train_loss]
            results['models_test_loss'] += [test_loss]
            results['models_start_loss'] += [start_loss]
            results['models_overfitted_state'] += [
                deepcopy(model.state_dict())
            ]
            results['models_loss'] += [fitted_state['loss']]
            results['models_fitted_state'] += [fitted_state['state_dict']]
            model.load_state_dict(fitted_state['state_dict'])
            results['models'] += [model]
            results['models_parameters'] += [model.get_total_parameters()]
            results['models_train_time'] += [train_time]
    array_sorted = numpy.argsort(array(results['models_loss']))
    for label in label_list:
        results[label] = [results[label][it] for it in array_sorted]
    return results
