"""
Modulo que contiene la implemtación general de una
arquitectura de red neuronal en Pytorch
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Any
from collections.abc import Callable
import numpy
from numpy import ndarray
import torch
from torch import Tensor
from torch.nn import (
    Sequential, Module, Linear, BatchNorm1d, Dropout, ReLU, ReLU6, PReLU,
    LeakyReLU, Hardsigmoid, Hardtanh, Tanh, Sigmoid, LogSoftmax, Softmax,
    Softmin, Softplus, Mish, SiLU, ELU, GELU, CELU, RReLU, SELU, Hardswish,
    Identity, LogSigmoid, Softsign, ModuleList, ConstantPad1d,
    Parameter
)
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.nn.init import (
    xavier_normal_, xavier_uniform_, kaiming_uniform_, uniform_,
    trunc_normal_, orthogonal_, sparse_, kaiming_normal_, normal_
)
from torch.nn.utils import fuse_linear_bn_weights


def gen_linear_layer(layer: int, capacity: list[int]):
    """
    A
    """
    assert capacity[-1] > 0, (
        "La capacidad final debe ser mayor a cero"
    )
    assert 0 <= layer and layer <= len(capacity) - 1, (
        f"El numero de capa debe estar entre 0 y {len(capacity) - 1}"
    )
    return (
        Linear(
            capacity[layer],
            next(cap for cap in capacity[(layer + 1):] if cap > 0),
            bias=False
        )
        if capacity[layer] > 0
        else Identity()
    )


class LinealNN(Module):
    """
    Clase que implementa una red neuronal general junto con su metodo
    de entrenamiento de propagacion hacia atras
    """
    functional_dict: dict[str, type[Module]] = {
        'ReLU': ReLU,
        'LeakyReLU': LeakyReLU,
        'ReLU6': ReLU6,
        'Hardsigmoid': Hardsigmoid,
        'Hardtanh': Hardtanh,
        'PReLU': PReLU,
        'Sigmoid': Sigmoid,
        'Softmin': Softmin,
        'Softmax': Softmax,
        'LogSoftmax': LogSoftmax,
        'Tanh': Tanh,
        'Softsign': Softsign,
        'Softplus': Softplus,
        'Mish': Mish,
        'SiLU': SiLU,
        'GELU': GELU,
        'CELU': CELU,
        'RReLU': RReLU,
        'SELU': SELU,
        'LogSigmoid': LogSigmoid,
        'Hardswish': Hardswish,
        'ELU': ELU,
        'None': Identity
    }
    init_dict: dict[str, Any] = {
        'xavier uniform': xavier_uniform_,
        'xavier normal': xavier_normal_,
        'kaiming uniform': kaiming_uniform_,
        'kaiming normal': kaiming_normal_,
        'uniform': uniform_,
        'normal': normal_,
        'trunc normal': trunc_normal_,
        'sparse': sparse_,
        'orthogonal': orthogonal_
    }
    layers: int
    capacity: list[int]
    bias: list[float]
    bias_layers: ModuleList
    dropout: list[float | None]
    dropout_layers: ModuleList
    batch_norm: list[bool]
    batch_norm_layers: ModuleList
    linear_layers: ModuleList
    activation: list[tuple[str, Any, dict]]
    activation_layers: ModuleList
    sequential_layers: ModuleList
    optimizer: Optimizer | None
    scheduler: LRScheduler | None
    loss_layer: Module
    inference_layer: Module
    log: dict[str, Any]
    activations: list[Tensor | None]
    masks: list[Tensor]
    weights_initializers: list[tuple[Any, Any, dict] | None]
    l1_weight: list[float | None]
    l1_activation: list[float | None]
    l2_weight: list[float | None]
    l2_activation: list[float | None]
    connection_dropout: list[float | None]
    verbosity: int
    overfit_tolerance: int | None
    seed: int | None
    learnable_layers: list[bool]

    def __init__(self):
        super().__init__()
        self.seed = None
        self.layers = -1
        self.verbosity = 0
        self.overfit_tolerance = None
        self.capacity = []
        self.bias = []
        self.dropout = []
        self.batch_norm = []
        self.activation = []
        self.masks = []
        self.weights_initializers = []
        self.activations = []
        self.l1_weight = []
        self.l2_weight = []
        self.l1_activation = []
        self.l2_activation = []
        self.connection_dropout = []
        self.log = {}
        self.loss_layer = Identity()
        self.inference_layer = Identity()
        self.optimizer = None
        self.scheduler = None
        self.__initialize_layers(capacity=[])

    @classmethod
    def from_capacity(cls, capacity: list[int], **kwargs):
        """
        A
        """
        assert capacity[-1] > 0 and capacity[0] > 0, (
            "La capacidad final e inicial deben ser mayor a cero"
        )
        model = cls()
        model.capacity = capacity
        layers = len(capacity) - 1
        if 'overfit_tolerance' in kwargs:
            model.overfit_tolerance = kwargs['overfit_tolerance']
        if 'inference_layer' in kwargs:
            if isinstance(kwargs['inference_layer'], Module):
                model.inference_layer = kwargs['inference_layer']
            else:
                model.inference_layer = InferenceModule(
                    kwargs['inference_layer']
                )
        if 'loss_layer' in kwargs:
            if isinstance(kwargs['loss_layer'], Module):
                model.loss_layer = kwargs['loss_layer']
            else:
                model.loss_layer = LossModule(kwargs['loss_layer'])
        model.learnable_layers = [True for _ in range(layers)]
        model.bias = [  # type: ignore
            1.0 if 'bias' not in kwargs
            else kwargs['bias'][k]
            if (
                isinstance(kwargs['bias'], list)
                or isinstance(kwargs['bias'], dict)
                and k in kwargs['bias']
            )
            else kwargs['bias']
            if (
                not isinstance(kwargs['bias'], dict)
                and kwargs['bias'] is not None
            )
            else 1.0
            for k in range(layers)
        ]
        model.dropout = [  # type: ignore
            None if 'dropout' not in kwargs
            else kwargs['dropout'][k]
            if (
                isinstance(kwargs['dropout'], list)
                or isinstance(kwargs['dropout'], dict)
                and k in kwargs['dropout']
            )
            else kwargs['dropout']
            if not isinstance(kwargs['dropout'], dict)
            else None
            for k in range(layers)
        ]
        model.connection_dropout = [  # type: ignore
            None if 'connection_dropout' not in kwargs
            else kwargs['connection_dropout'][k]
            if (
                isinstance(kwargs['connection_dropout'], list)
                or isinstance(kwargs['connection_dropout'], dict)
                and k in kwargs['connection_dropout']
            )
            else kwargs['connection_dropout']
            if not isinstance(kwargs['connection_dropout'], dict)
            else None
            for k in range(layers)
        ]
        model.batch_norm = [  # type: ignore
            False if 'batch_norm' not in kwargs
            else kwargs['batch_norm'][k]
            if (
                isinstance(kwargs['batch_norm'], list)
                or isinstance(kwargs['batch_norm'], dict)
                and k in kwargs['batch_norm']
            )
            else kwargs['batch_norm']
            if (
                not isinstance(kwargs['batch_norm'], dict)
                and kwargs['batch_norm'] is not None
            )
            else False
            for k in range(layers)
        ]
        model.activation = [  # type: ignore
            ('None',) if 'activation' not in kwargs
            else kwargs['activation'][k]
            if (
                isinstance(kwargs['activation'], list)
                or isinstance(kwargs['activation'], dict)
                and k in kwargs['activation']
            )
            else kwargs['activation']
            if (
                not isinstance(kwargs['activation'], dict)
                and kwargs['activation'] is not None
            )
            else ('None',)
            for k in range(layers)
        ]
        model.weights_initializers = [  # type: ignore
            None if 'weights_initializers' not in kwargs
            else kwargs['weights_initializers'][k]
            if (
                isinstance(kwargs['weights_initializers'], list)
                or isinstance(kwargs['weights_initializers'], dict)
                and k in kwargs['weights_initializers']
            )
            else kwargs['weights_initializers']
            if not isinstance(kwargs['weights_initializers'], dict)
            else None
            for k in range(layers)
        ]
        model.masks = [
            torch.full(
                (capacity[k + 1], capacity[k] + 1),
                1
            ).bool()
            if 'masks' not in kwargs
            else kwargs['masks'][k]
            if (
                isinstance(kwargs['masks'], list)
                or isinstance(kwargs['masks'], dict)
                and k in kwargs['masks']
            )
            else torch.full(
                (capacity[k + 1], capacity[k] + 1),
                1
            ).bool()
            for k in range(layers)
        ]
        model.l1_weight = [  # type: ignore
            None if 'l1_weight' not in kwargs
            else kwargs['l1_weight'][k]
            if (
                isinstance(kwargs['l1_weight'], list)
                or isinstance(kwargs['l1_weight'], dict)
                and k in kwargs['l1_weight']
            )
            else kwargs['l1_weight']
            if not isinstance(kwargs['l1_weight'], dict)
            else None
            for k in range(layers)
        ]
        model.l1_activation = [  # type: ignore
            None if 'l1_activation' not in kwargs
            else kwargs['l1_activation'][k]
            if (
                isinstance(kwargs['l1_activation'], list)
                or isinstance(kwargs['l1_activation'], dict)
                and k in kwargs['l1_activation']
            )
            else kwargs['l1_activation']
            if not isinstance(kwargs['l1_activation'], dict)
            else None
            for k in range(layers)
        ]
        model.l2_weight = [  # type: ignore
            None if 'l2_weight' not in kwargs
            else kwargs['l2_weight'][k]
            if (
                isinstance(kwargs['l2_weight'], list)
                or isinstance(kwargs['l2_weight'], dict)
                and k in kwargs['l2_weight']
            )
            else kwargs['l2_weight']
            if not isinstance(kwargs['l2_weight'], dict)
            else None
            for k in range(layers)
        ]
        model.l2_activation = [  # type: ignore
            None if 'l2_activation' not in kwargs
            else kwargs['l2_activation'][k]
            if (
                isinstance(kwargs['l2_activation'], list)
                or isinstance(kwargs['l2_activation'], dict)
                and k in kwargs['l2_activation']
            )
            else kwargs['l2_activation']
            if not isinstance(kwargs['l2_activation'], dict)
            else None
            for k in range(layers)
        ]
        model.verbosity = (
            kwargs['verbose'].count('v')
            if isinstance(kwargs['verbose'], str)
            else 0
        )
        model.__initialize_layers(capacity=capacity)
        return model

    def set_masks(self, masks: list[Tensor | None] | dict[int, Tensor | None]):
        """
        A
        """
        self.masks = [  # type: ignore
            (
                torch.full(
                    (self.capacity[k + 1], self.capacity[k] + 1),
                    1
                ).bool()
                if masks[k] is None
                else masks[k]
            )
            if (
                isinstance(masks, list)
                or isinstance(masks, dict)
                and k in masks
            )
            else torch.full(
                (self.capacity[k + 1], self.capacity[k] + 1),
                1
            ).bool()
            for k in range(self.layers)
        ]
        self.__update_masks()

    def __initialize_layers(self, capacity: list[int]):
        layers = len(capacity) - 1
        self.bias_layers = ModuleList([
            ConstantPad1d((0, 1), 1) for _ in range(layers)
        ])
        self.dropout_layers = ModuleList([
            Dropout(p=self.dropout[k], inplace=True)  # type: ignore
            if self.dropout[k] is not None
            else Identity()
            for k in range(layers)
        ])
        self.batch_norm_layers = ModuleList([
            BatchNorm1d(self.capacity[k])
            if self.batch_norm[k]
            else Identity()
            for k in range(layers)
        ])
        self.linear_layers = ModuleList([
            gen_linear_layer(k, self.capacity)
            for k in range(layers)
        ])
        self.activation_layers = ModuleList([
            self.functional_dict[self.activation[k][0]](
                *self.activation[k][1],
                **self.activation[k][2]
            )
            for k in range(layers)
        ])
        self.sequential_layers = ModuleList([
            Sequential(
                self.bias_layers[k],
                self.dropout_layers[k],
                self.batch_norm_layers[k],
                self.linear_layers[k],
                self.activation_layers[k]
            )
            for k in range(layers)
        ])
        self.activations = [None for _ in range(layers)]
        for linear_layer, initializer in zip(
            self.linear_layers,
            self.weights_initializers
        ):
            if isinstance(linear_layer, Linear):
                if initializer is not None:
                    initializer[0](
                        linear_layer.weight,
                        *initializer[1],
                        **initializer[2]
                    )
        self.layers = layers
        self.__update_masks()
        self.__upgdate_learnable_layers()

    def __update_masks(self):
        """
        A
        """
        for mask, linear_layer in zip(
            self.masks,
            self.linear_layers
        ):
            if isinstance(linear_layer, Linear):
                linear_layer.weight.data.copy_(
                    linear_layer.weight * mask.type_as(  # type: ignore
                        linear_layer.weight  # type: ignore
                    )
                )

    def set_optimizer(self, optimizer: type[Optimizer], *args, **kwargs):
        """
        A
        """
        self.optimizer = optimizer(self.parameters(), *args, **kwargs)
        self.scheduler = None

    def set_scheduler(self, scheduler: type[LRScheduler], *args, **kwargs):
        """
        A
        """
        if self. optimizer is not None:
            self.scheduler = scheduler(self.optimizer, *args, **kwargs)

    def __upgdate_learnable_layers(self):
        for sequential_layer, learnable in zip(
            self.sequential_layers, self.learnable_layers
        ):
            sequential_layer.requires_grad_(learnable)

    def forward(self, input_tensor: Tensor) -> Tensor:
        """
        Implementacion necesaria para la implementacion de Module en Pytorch,
        es la evaluación hacia delante
        """
        x = input_tensor
        for k, layer in enumerate(self.sequential_layers):
            x = layer(x)
            self.activations[k] = x
        return x

    def __log_loss(self, label: str, loss: float):
        print(f'{label:30} loss : {loss:>10g}')

    def __log_start(self):
        match self.verbosity:
            case 3:
                self.__log_loss(
                    'start       test',
                    self.log['test loss']
                )
            case 2:
                self.__log_loss(
                    'start       test',
                    self.log['test loss']
                )
            case 1:
                self.__log_loss(
                    'start       test',
                    self.log['test loss']
                )
            case _:
                pass

    def __log_epoch(self, epoch: int):
        match self.verbosity:
            case 3:
                self.__log_loss(
                    f'epoch {epoch:>5d} train    raw',
                    self.log['raw loss']
                )
                self.__log_loss(
                    f'epoch {epoch:>5d} train normalized',
                    self.log['norm loss']
                )
                self.__log_loss(
                    f'epoch {epoch:>5d} test',
                    self.log['test loss']
                )
            case 2:
                self.__log_loss(
                    f'epoch {epoch:>5d} train normalized',
                    self.log['norm loss']
                )
                self.__log_loss(
                    f'epoch {epoch:>5d} test',
                    self.log['test loss']
                )
            case 1:
                self.__log_loss(
                    f'epoch {epoch:>5d} test',
                    self.log['test loss']
                )
            case _:
                pass

    def l2_weight_regularization(self):
        """
        A
        """
        l2_layers = [
            (
                norm_weight *
                linear_layer.weight.square().mean()  # type: ignore
            ).unsqueeze(dim=0)
            for linear_layer, norm_weight in zip(
                self.linear_layers,
                self.l2_weight
            ) if norm_weight is not None
        ]
        if len(l2_layers) > 0:
            return torch.concat(l2_layers).mean()
        return torch.tensor(0)

    def l1_activation_regularization(self):
        """
        A
        """
        l1_layers = [
            (
                norm_weight
                * activation.abs().mean()  # type: ignore
            ).unsqueeze(dim=0)
            for activation, norm_weight in zip(
                self.activations,
                self.l1_activation
            ) if norm_weight is not None
        ]
        if len(l1_layers) > 0:
            return torch.concat(l1_layers).mean()
        return torch.tensor(0)

    def l2_activation_regularization(self):
        """
        A
        """
        l2_layers = [
            (
                norm_weight
                * activation.square().mean()  # type: ignore
            ).unsqueeze(dim=0)
            for activation, norm_weight in zip(
                self.activations,
                self.l2_activation
            ) if norm_weight is not None
        ]
        if len(l2_layers) > 0:
            return torch.concat(l2_layers).mean()
        return torch.tensor(0)

    def l1_weight_regularization(self):
        """
        A
        """
        l1_layers = [
            (
                norm_weight
                * linear_layer.weight.abs().mean()  # type: ignore
            ).unsqueeze(dim=0)
            for linear_layer, norm_weight in zip(
                self.linear_layers,
                self.l1_weight
            ) if norm_weight is not None
        ]
        if len(l1_layers) > 0:
            return torch.concat(l1_layers).mean()
        return torch.tensor(0)

    def train_batch(
        self, batch_features: Tensor,
        batch_targets: Tensor
    ):
        """
        A
        """
        def closure():
            with ThreadPoolExecutor(max_workers=10) as executor:
                self.train()
                weights_copy = []
                for linear_layer, connect_dropout, mask in (
                    (linear_layer, connect_dropout, mask)
                    for linear_layer, connect_dropout, mask in zip(
                        self.linear_layers,
                        self.connection_dropout,
                        self.masks
                    ) if connect_dropout is not None
                ):
                    weight = linear_layer.weight
                    connection_mask = torch.full_like(
                        weight,  # type: ignore
                        connect_dropout
                    ).bernoulli().bool()
                    weight_mask = mask.bitwise_and(connection_mask)
                    linear_layer.weight = (
                        weight.clone() * weight_mask.type_as(  # type: ignore
                            weight  # type: ignore
                        )
                    )
                    weights_copy += [weight]
                if self.optimizer is not None:
                    self.optimizer.zero_grad()
                loss = self.loss_layer(self(batch_features), batch_targets)
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
                self.log['batch raw loss'] = loss.cpu().detach().item()
                loss += (
                    l1w_future.result() + l2w_future.result() +
                    l1a_future.result() + l2a_future.result()
                )
                self.log['batch norm loss'] = loss.cpu().detach().item()
                loss.backward()
                for weight, (linear_layer, connect_dropout, mask) in zip(
                    weights_copy,
                    (
                        (linear_layer, connect_dropout, mask)
                        for linear_layer, connect_dropout, mask in zip(
                            self.linear_layers,
                            self.connection_dropout,
                            self.masks
                        ) if connect_dropout is not None
                    )
                ):
                    linear_layer.weight = weight  # type: ignore
                for linear_layer, mask in zip(
                    self.linear_layers, self.masks
                ):
                    linear_layer.weight.grad.copy_(  # type: ignore
                        linear_layer.weight.grad
                        * mask.type_as(  # type: ignore
                            linear_layer.weight.grad  # type: ignore
                        )
                    )
                self.eval()
                return loss.cpu().detach().item()
        if self.optimizer is not None:
            return self.optimizer.step(closure=closure)
        return closure()

    def train_epoch(self, dataloader: DataLoader):
        """
        A
        """
        batch_loss = []
        self.log['norm loss'] = []
        self.log['raw loss'] = []
        for features, targets in dataloader:
            batch_loss += [self.train_batch(features, targets)]
            self.log['norm loss'] += [self.log['batch norm loss']]
            self.log['raw loss'] += [self.log['batch raw loss']]
        if self.scheduler is not None:
            self.scheduler.step()
        self.log['raw loss'] = numpy.mean(self.log['raw loss']).item()
        self.log['norm loss'] = numpy.mean(self.log['norm loss']).item()
        return numpy.mean(batch_loss).item()

    def test_epoch(self, dataloader: DataLoader):
        """
        A
        """
        with torch.inference_mode():
            loss = numpy.mean([
                self.loss_layer(self(X), y).cpu().detach().item()
                for X, y in dataloader
            ]).item()
            self.log['test loss'] = loss
        return loss

    def train_loop(
        self,
        epochs: int,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader | None = None,
        seed: int | None = None
    ):
        """
        A
        """
        if seed is None:
            seed = torch.seed()
        else:
            torch.manual_seed(seed)
        self.seed = seed
        if test_dataloader is None:
            test_dataloader = train_dataloader
        train_loss, test_loss = [], []
        start_loss = self.test_epoch(test_dataloader)
        self.__log_start()
        fitted_state = {'state_dict': {}, 'loss': None, 'epoch': None}
        overfit_counter = self.overfit_tolerance
        for epoch in range(epochs):
            train_loss += [self.train_epoch(train_dataloader)]
            loss = self.test_epoch(test_dataloader)
            test_loss += [loss]
            self.__log_epoch(epoch)
            if self.overfit_tolerance is not None:
                overfit_counter -= 1  # type: ignore
                if fitted_state['loss'] is None or loss < fitted_state['loss']:
                    overfit_counter = self.overfit_tolerance
                    fitted_state['epoch'] = epoch
                    fitted_state['loss'] = loss
                    fitted_state['state_dict'] = deepcopy(self.state_dict())
                if overfit_counter < 0:
                    break
        return train_loss, test_loss, start_loss, fitted_state

    def inference(self, x: Tensor):
        """
        Ciclo principal para obtener solamente la inferencia de la red neuronal
        sobre un conjunto de datos
        """
        with torch.inference_mode():
            return self.inference_layer(
                self(x)
            ).cpu().detach().numpy()

    def test_loop(self, dataloader: DataLoader):
        """
        Ciclo principal para calcular unicamente la perdida
        de los datos de prueba
        """
        with torch.inference_mode():
            return self.test_epoch(dataloader)

    def get_weights(self):
        """
        Metodo que devuelve una lista de arreglos multidimesionales que
        contiene los pesos de las capas lineales en una red neuronal
        """
        with torch.inference_mode():
            weights = []
            for k, linear in enumerate(self.linear_layers):
                w, b = torch.tensor_split(
                    linear.weight,  # type: ignore
                    (self.capacity[k], 1),
                    dim=1
                )
                if isinstance(self.batch_norm_layers[k], BatchNorm1d):
                    bn = self.batch_norm_layers[k]
                    w, b = fuse_linear_bn_weights(
                        w, b, bn.running_mean, bn.running_var,  # type: ignore
                        bn.eps, bn.weight, bn.bias  # type: ignore
                    )
                weight = torch.concat((w, b), dim=1)
                weights += [weight.cpu().detach().numpy()]
        return weights

    def set_weights(self, weights: list[ndarray]):
        """
        Metodo que devuelve una lista de arreglos multidimesionales
        que contiene los pesos de las capas lineales en una red neuronal
        """
        for k in range(self.layers):
            if isinstance(self.linear_layers[k], Linear):
                if self.batch_norm[k]:
                    self.batch_norm_layers[k] = BatchNorm1d(
                        self.capacity[k] + 1
                    )
                self.linear_layers[k].weight = Parameter(  # type: ignore
                    torch.tensor(weights[k], requires_grad=True)
                    * self.masks[k].type_as(
                        weights[k]  # type: ignore
                    )
                )

    def get_total_parameters(self):
        """
        A
        """
        return numpy.sum([
            linear.weight.numel()  # type: ignore
            for linear in self.linear_layers
        ]).item()

    def copy_with_masks(self, masks: list[Tensor]):
        """
        A
        """
        arch_copy = LinealNN()
        arch_copy.load_state_dict(deepcopy(self.state_dict()), assign=True)
        arch_copy.set_masks(masks)  # type: ignore
        return arch_copy

    def set_learnable_layers(self, learnable: list[bool]):
        """
        A
        """
        self.learnable_layers = learnable
        self.__upgdate_learnable_layers()

    def enable_learnable_layer(self, layer: int):
        """
        A
        """
        self.learnable_layers[layer] = True
        self.__upgdate_learnable_layers()

    def disable_learnable_layer(self, layer: int):
        """
        A
        """
        self.learnable_layers[layer] = False
        self.__upgdate_learnable_layers()


class InferenceModule(Module):
    """
    A
    """
    inference_function: Callable[[Tensor], Tensor]

    def __init__(self, inference_function: Callable[[Tensor], Tensor]) -> None:
        self.inference_function = inference_function
        super().__init__()

    def forward(self, input_tensor: Tensor):
        """
        A
        """
        return self.inference_function(input_tensor)


class LossModule(Module):
    """
    A
    """
    loss_function: Callable[[Tensor, Tensor], Tensor]

    def __init__(
        self,
        loss_function: Callable[[Tensor, Tensor], Tensor]
    ) -> None:
        self.loss_function = loss_function
        super().__init__()

    def forward(self, input_a: Tensor, input_b: Tensor):
        """
        A
        """
        return self.loss_function(input_a, input_b)
