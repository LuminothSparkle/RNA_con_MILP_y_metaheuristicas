"""
Modulo que contiene la implemtación general de una
arquitectura de red neuronal en Pytorch
"""
from copy import deepcopy
from typing import Any
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

def process_layer_param(param: Any, layers: int):
    return [
        param if (
            not isinstance(param, list) and
            not isinstance(param, dict)
        )
        else param[k] if (
            isinstance(param, list)
            or isinstance(param, dict) and k in param
        )
        else param[str(k)]
        if isinstance(param, dict) and str(k) in param
        else None
        for k in range(layers)
    ]


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
    activation_layers: ModuleList
    sequential_layers: ModuleList
    bias_layers: ModuleList
    dropout_layers: ModuleList
    batch_norm_layers: ModuleList
    linear_layers: ModuleList
    activations: list[Tensor | None]
    masks_layer: list[Tensor | None]
    layers: int
    capacity: list[int]
    masks: list[Tensor | None] | dict[str|int, Tensor | None] | Tensor | None
    bias: list[float | None] | dict[str|int, float | None] | float | None
    dropout: list[float | None] | dict[str|int, float | None] | float | None
    batch_norm: list[bool | None] | dict[str|int, bool | None] | bool | None
    activation: list[tuple[str, Any, dict]  | None] | dict[str|int, tuple[str, Any, dict] | None] | tuple[str, Any, dict] | None
    weights_initializers: list[tuple[Any, Any, dict]  | None] | dict[str|int, tuple[Any, Any, dict]  | None] | tuple[Any, Any, dict]  | None
    learnable_layers: list[bool  | None] | dict[str|int, bool | None] | bool | None

    def __init__(
        self, capacity: list[int] | None = None,
        masks: list[Tensor | None] | dict[str|int, Tensor | None] | Tensor | None = None,
        bias: list[float | None] | dict[str|int, float | None] | float | None = None,
        dropout: list[float | None] | dict[str|int, float | None] | float | None = None,
        batch_norm: list[bool | None] | dict[str|int, bool | None] | bool | None = None,
        activation: list[tuple[str, Any, dict]  | None] | dict[str|int, tuple[str, Any, dict] | None] | tuple[str, Any, dict] | None = None,
        weights_initializers: list[tuple[Any, Any, dict]  | None] | dict[str|int, tuple[Any, Any, dict]  | None] | tuple[Any, Any, dict]  | None = None,
        learnable_layers: list[bool  | None] | dict[str|int, bool | None] | bool | None = None
    ):
        super().__init__()
        self.start(
            capacity=capacity, masks=masks, bias=bias, dropout=dropout, batch_norm=batch_norm,
            activation=activation, weights_initializers=weights_initializers,
            learnable_layers=learnable_layers
        )

    def start(self, capacity: list[int] | None = None,
        masks: list[Tensor | None] | dict[str|int, Tensor | None] | Tensor | None = None,
        bias: list[float | None] | dict[str|int, float | None] | float | None = None,
        dropout: list[float | None] | dict[str|int, float | None] | float | None = None,
        batch_norm: list[bool | None] | dict[str|int, bool | None] | bool | None = None,
        activation: list[tuple[str, Any, dict]  | None] | dict[str|int, tuple[str, Any, dict] | None] | tuple[str, Any, dict] | None = None,
        weights_initializers: list[tuple[Any, Any, dict]  | None] | dict[str|int, tuple[Any, Any, dict]  | None] | tuple[Any, Any, dict]  | None = None,
        learnable_layers: list[bool  | None] | dict[str|int, bool | None] | bool | None = None
    ):
        torchdefault.set_defaults()
        if capacity is None:
            capacity = []
        self.bias = bias
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.activation = activation
        self.weights_initializers = weights_initializers
        self.learnable_layers = learnable_layers
        self.set_capacity(capacity=capacity)
        self.set_masks(masks)

    def state_dict(self, keep_vars=True, **kwargs):
        return {'init': {
            'capacity': self.capacity,
            'bias': self.bias,
            'dropout': self.dropout,
            'batch_norm': self.batch_norm,
            'masks': self.masks,
            'weights_initializers': self.weights_initializers,
            'learnable_layers': self.learnable_layers
        }} | super().state_dict(keep_vars=keep_vars, **kwargs)

    def load_state_dict(self, state_dict: dict, strict = True, assign = True):
        new_state_dict = deepcopy(state_dict)
        init = new_state_dict.pop('init')
        self.start(**init)
        r = super().load_state_dict(state_dict=new_state_dict, strict=strict, assign=assign)
        return r

    def set_masks(
        self,
        masks: list[Tensor | None] | dict[int, Tensor | None] | None = None
    ):
        """
        A
        """
        torchdefault.set_defaults()
        self.masks = masks
        self.masks_layer = [  # type: ignore
            torch.ones((
                next(
                    cap for cap in self.capacity[(k + 1):]
                    if cap > 0
                ),
                self.capacity[k] + 1
            )).bool()
            if mask is None
            else mask
            for k, mask in zip(
                range(self.layers),
                process_layer_param(self.masks, self.layers)
            )
        ]
        self.__update_masks()

    def set_capacity(self, capacity: list[int]):
        torchdefault.set_defaults()
        self.capacity = capacity
        self.layers = len(capacity) - 1
        self.bias_layers = ModuleList([
            ConstantPad1d((0, 1), param)
            if param is not None else ConstantPad1d((0, 1), 1)
            for param in process_layer_param(self.bias, self.layers)
        ])
        self.dropout_layers = ModuleList([
            Dropout(p=param, inplace=True)
            if param is not None else Identity()
            for param in process_layer_param(self.dropout, self.layers)
        ])
        self.batch_norm_layers = ModuleList([
            BatchNorm1d(self.capacity[k])
            if param is not None and param else Identity()
            for k, param in zip(
                range(self.layers),
                process_layer_param(self.batch_norm, self.layers)
            )
        ])
        self.linear_layers = ModuleList([
            gen_linear_layer(k, self.capacity)
            for k in range(self.layers)
        ])
        self.activation_layers = ModuleList([
            self.functional_dict[param[0]](*param[1],**param[2])
            if param is not None and param else Identity()
            for param in process_layer_param(self.activation, self.layers)
        ])
        self.sequential_layers = ModuleList([
            Sequential(
                self.bias_layers[k],
                self.dropout_layers[k],
                self.batch_norm_layers[k],
                self.linear_layers[k],
                self.activation_layers[k]
            )
            for k in range(self.layers)
        ])
        self.activations = [None for _ in range(self.layers)]
        for linear_layer, initializer in zip(
            self.linear_layers,
            process_layer_param(self.weights_initializers, self.layers)
        ):
            if isinstance(linear_layer, Linear):
                if initializer is not None:
                    self.init_dict[initializer[0]](
                        linear_layer.weight,
                        *initializer[1],
                        **initializer[2]
                    )
        self.__update_learnable_layers()

    def __update_masks(self):
        """
        A
        """
        torchdefault.set_defaults()
        for mask, linear_layer in zip(self.masks_layer, self.linear_layers):
            if isinstance(linear_layer, Linear) and mask is not None:
                linear_layer.weight.data.copy_(
                    linear_layer.weight * mask.type_as(  # type: ignore
                        linear_layer.weight  # type: ignore
                    )
                )

    def __update_learnable_layers(self):
        for sequential_layer, learnable in zip(
            self.sequential_layers,
            process_layer_param(self.learnable_layers, self.layers)
        ):
            sequential_layer.requires_grad_(learnable if learnable is not None else True)

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
        for k, mask, batch_norm in zip(
            range(self.layers), self.masks_layer,
            process_layer_param(self.batch_norm, self.layers)
        ):
            if isinstance(self.linear_layers[k], Linear):
                if batch_norm:
                    self.batch_norm_layers[k] = BatchNorm1d(
                        self.capacity[k] + 1
                    )
                if mask is not None:
                    self.linear_layers[k].weight = Parameter(  # type: ignore
                        torchdefault.tensor(weights[k], requires_grad=True)
                        * mask.to(
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
