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
from torch.nn.init import (
    xavier_normal_, xavier_uniform_, kaiming_uniform_, uniform_,
    trunc_normal_, orthogonal_, sparse_, kaiming_normal_, normal_
)
from torch.nn.utils import fuse_linear_bn_weights
import utility.nn.torchdefault as torchdefault

def gen_linear_layer(layer: int, capacity: list[int]):
    """
    A
    """
    torchdefault.set_defaults()
    assert capacity[-1] > 0, (
        "La capacidad final debe ser mayor a cero"
    )
    assert 0 <= layer and layer <= len(capacity) - 1, (
        f"El numero de capa debe estar entre 0 y {len(capacity) - 1}"
    )
    return (
        Linear(
            capacity[layer] + 1,
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
    loss_layer: Module
    inference_layer: Module
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
        torchdefault.set_defaults()
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
        self.learnable_layers = []
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
        torchdefault.set_defaults()
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
            1.0 if (
                'bias' not in kwargs
                or isinstance(kwargs['bias'], dict)
                and k not in kwargs['bias']
            )
            else kwargs['bias'][k]
            if (
                isinstance(kwargs['bias'], list)
                or isinstance(kwargs['bias'], dict)
                and k in kwargs['bias']
            )
            else kwargs['bias']
            for k in range(layers)
        ]
        model.dropout = [  # type: ignore
            None if (
                'dropout' not in kwargs
                or isinstance(kwargs['dropout'], dict)
                and k not in kwargs['dropout']
            )
            else kwargs['dropout'][k]
            if (
                isinstance(kwargs['dropout'], list)
                or isinstance(kwargs['dropout'], dict)
                and k in kwargs['dropout']
            )
            else kwargs['dropout']
            for k in range(layers)
        ]
        model.connection_dropout = [  # type: ignore
            None if (
                'connection_dropout' not in kwargs
                or isinstance(kwargs['connection_dropout'], dict)
                and k not in kwargs['connection_dropout']
            )
            else kwargs['connection_dropout'][k]
            if (
                isinstance(kwargs['connection_dropout'], list)
                or isinstance(kwargs['connection_dropout'], dict)
                and k in kwargs['connection_dropout']
            )
            else kwargs['connection_dropout']
            for k in range(layers)
        ]
        model.batch_norm = [  # type: ignore
            False if (
                'batch_norm' not in kwargs
                or isinstance(kwargs['batch_norm'], dict)
                and k not in kwargs['batch_norm']
            )
            else kwargs['batch_norm'][k]
            if (
                isinstance(kwargs['batch_norm'], list)
                or isinstance(kwargs['batch_norm'], dict)
                and k in kwargs['batch_norm']
            )
            else kwargs['batch_norm']
            for k in range(layers)
        ]
        model.activation = [  # type: ignore
            ('None',) if (
                'activation' not in kwargs
                or isinstance(kwargs['activation'], dict)
                and k not in kwargs['activation']
            )
            else kwargs['activation'][k]
            if (
                isinstance(kwargs['activation'], list)
                or isinstance(kwargs['activation'], dict)
                and k in kwargs['activation']
            )
            else kwargs['activation']
            for k in range(layers)
        ]
        model.weights_initializers = [  # type: ignore
            None if (
                'weights_initializers' not in kwargs
                or isinstance(kwargs['weights_initializers'], dict)
                and k not in kwargs['weights_initializers']
            )
            else kwargs['weights_initializers'][k]
            if (
                isinstance(kwargs['weights_initializers'], list)
                or isinstance(kwargs['weights_initializers'], dict)
                and k in kwargs['weights_initializers']
            )
            else kwargs['weights_initializers']
            for k in range(layers)
        ]
        model.l1_weight = [  # type: ignore
            None if (
                'l1_weight' not in kwargs
                or k + 1 >= layers
                or isinstance(kwargs['l1_weight'], dict)
                and k not in kwargs['l1_weight']
            )
            else kwargs['l1_weight'][k]
            if (
                isinstance(kwargs['l1_weight'], list)
                or isinstance(kwargs['l1_weight'], dict)
                and k in kwargs['l1_weight']
            )
            else kwargs['l1_weight']
            for k in range(layers)
        ]
        model.l1_activation = [  # type: ignore
            None if (
                'l1_activation' not in kwargs
                or k + 1 >= layers
                or isinstance(kwargs['l1_activation'], dict)
                and k not in kwargs['l1_activation']
            )
            else kwargs['l1_activation'][k]
            if (
                isinstance(kwargs['l1_activation'], list)
                or isinstance(kwargs['l1_activation'], dict)
                and k in kwargs['l1_activation']
            )
            else kwargs['l1_activation']
            for k in range(layers)
        ]
        model.l2_weight = [  # type: ignore
            None if (
                'l2_weight' not in kwargs
                or k + 1 >= layers
                or isinstance(kwargs['l2_weight'], dict)
                and k not in kwargs['l2_weight']
            )
            else kwargs['l2_weight'][k]
            if (
                isinstance(kwargs['l2_weight'], list)
                or isinstance(kwargs['l2_weight'], dict)
                and k in kwargs['l2_weight']
            )
            else kwargs['l2_weight']
            for k in range(layers)
        ]
        model.l2_activation = [  # type: ignore
            None if (
                'l2_activation' not in kwargs
                or k + 1 >= layers
                or isinstance(kwargs['l2_activation'], dict)
                and k not in kwargs['l2_activation']
            )
            else kwargs['l2_activation'][k]
            if (
                isinstance(kwargs['l2_activation'], list)
                or isinstance(kwargs['l2_activation'], dict)
                and k in kwargs['l2_activation']
            )
            else kwargs['l2_activation']
            for k in range(layers)
        ]
        model.verbosity = (
            kwargs['verbose'].count('v')
            if isinstance(kwargs['verbose'], str)
            else 0
        )
        model.__initialize_layers(capacity=capacity)
        model.set_masks(kwargs['masks'] if 'masks' in kwargs else None)
        return model

    def set_capacity(self, capacity: list[int]):
        self.__initialize_layers(capacity=capacity)

    def set_masks(
        self,
        masks: list[Tensor | None] | dict[int, Tensor | None] | None = None
    ):
        """
        A
        """
        torchdefault.set_defaults()
        self.masks = [  # type: ignore
            (
                torch.ones((
                    next(
                        cap for cap in self.capacity[(k + 1):]
                        if cap > 0
                    ),
                    self.capacity[k] + 1
                )).bool()
                if masks[k] is None
                else masks[k]
            )
            if (
                isinstance(masks, list)
                or isinstance(masks, dict)
                and k in masks
            )
            else torch.ones((
                next(
                    cap for cap in self.capacity[(k + 1):]
                    if cap > 0
                ),
                self.capacity[k] + 1
            )).bool()
            for k in range(self.layers)
        ]
        self.__update_masks()

    def __initialize_layers(self, capacity: list[int]):
        torchdefault.set_defaults()
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
        self.__update_learnable_layers()

    def __update_masks(self):
        """
        A
        """
        torchdefault.set_defaults()
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

    def __update_learnable_layers(self):
        torchdefault.set_defaults()
        for sequential_layer, learnable in zip(
            self.sequential_layers, self.learnable_layers
        ):
            sequential_layer.requires_grad_(learnable)

    def forward(self, input_tensor: Tensor) -> Tensor:
        """
        Implementacion necesaria para la implementacion de Module en Pytorch,
        es la evaluación hacia delante
        """
        torchdefault.set_defaults()
        x = input_tensor
        for k, layer in enumerate(self.sequential_layers):
            x = layer(x)
            self.activations[k] = x
        return x

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
                self.linear_layers,
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
                self.activations,
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
                self.activations,
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
                self.linear_layers,
                self.l1_weight
            ) if norm_weight is not None
        ]
        if len(l1_layers) > 0:
            return torch.concat(l1_layers).mean()
        return torchdefault.tensor(0)

    def train_closure(self, features: Tensor, targets: Tensor):
        torchdefault.set_defaults()
        results = {}
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
                connect_dropout,
                dtype=torch.get_default_dtype(),
                device=torch.get_default_device()
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
        loss = self.loss_layer(self(features), targets)
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
            results['raw_loss'] = loss.cpu().detach().item()
            loss += (
                l1w_future.result() + l2w_future.result() +
                l1a_future.result() + l2a_future.result()
            )
            results['norm_loss'] = loss.cpu().detach().item()
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
        return results

    def inference(self, x: Tensor):
        """
        Ciclo principal para obtener solamente la inferencia de la red neuronal
        sobre un conjunto de datos
        """
        torchdefault.set_defaults()
        with torch.inference_mode():
            return self.inference_layer(
                self(x)
            ).cpu().detach().numpy()

    def loss(self, features: Tensor, target: Tensor):
        torchdefault.set_defaults()
        with torch.inference_mode():
            return self.loss_layer(
                self(features),
                target
            ).cpu().detach().numpy()

    def get_weights(self):
        """
        Metodo que devuelve una lista de arreglos multidimesionales que
        contiene los pesos de las capas lineales en una red neuronal
        """
        torchdefault.set_defaults()
        with torch.inference_mode():
            weights = []
            for k, linear in enumerate(self.linear_layers):
                if isinstance(linear, Linear):
                    w, b = torch.split_with_sizes(
                        linear.weight,  # type: ignore
                        (self.capacity[k], 1),
                        dim=1
                    )
                    if isinstance(self.batch_norm_layers[k], BatchNorm1d):
                        bn = self.batch_norm_layers[k]
                        w, b = fuse_linear_bn_weights(
                            w, b, bn.running_mean,  # type: ignore
                            bn.running_var,  # type: ignore
                            bn.eps, bn.weight, bn.bias  # type: ignore
                        )
                    weight = torch.concat((w, b), dim=1)
                    weights += [weight.cpu().detach().numpy()]
                else:
                    weights += [None]
        return weights

    def set_weights(self, weights: list[ndarray]):
        """
        Metodo que devuelve una lista de arreglos multidimesionales
        que contiene los pesos de las capas lineales en una red neuronal
        """
        torchdefault.set_defaults()
        for k in range(self.layers):
            if isinstance(self.linear_layers[k], Linear):
                if self.batch_norm[k]:
                    self.batch_norm_layers[k] = BatchNorm1d(
                        self.capacity[k] + 1
                    )
                self.linear_layers[k].weight = Parameter(  # type: ignore
                    torchdefault.tensor(weights[k], requires_grad=True)
                    * self.masks[k].to(
                        device=torch.get_default_device(),
                        dtype=torch.get_default_dtype()
                    )
                )
        self.__update_masks()

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
        arch_copy = self.copy()
        arch_copy.set_masks(masks)  # type: ignore
        return arch_copy

    def set_learnable_layers(self, learnable: list[bool]):
        """
        A
        """
        torchdefault.set_defaults()
        self.learnable_layers = learnable
        self.__update_learnable_layers()

    def enable_learnable_layer(self, layer: int):
        """
        A
        """
        torchdefault.set_defaults()
        self.learnable_layers[layer] = True
        self.__update_learnable_layers()

    def disable_learnable_layer(self, layer: int):
        """
        A
        """
        torchdefault.set_defaults()
        self.learnable_layers[layer] = False
        self.__update_learnable_layers()

    def copy(self):
        return deepcopy(self)


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
        torchdefault.set_defaults()
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
        torchdefault.set_defaults()
        return self.loss_function(input_a, input_b)
