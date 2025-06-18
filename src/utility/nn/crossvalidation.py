"""
Modulo que contiene los metodos para realizar validacion cruzada con scikit learn y pytorch
"""

from typing import Any
from abc import abstractmethod
from collections.abc import Iterable, Callable, Generator

from time import perf_counter_ns

import numpy
from numpy import array, ndarray
from numpy.typing import ArrayLike

from pandas import DataFrame, Index

from scipy.sparse import spmatrix

import torch
from torch.nn import Module
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset, TensorDataset, Subset
from torch.nn.functional import (cross_entropy, mse_loss,
                            binary_cross_entropy_with_logits)

from sklearn.model_selection import BaseCrossValidator
from sklearn.preprocessing import (LabelBinarizer, LabelEncoder, StandardScaler,
                                    OrdinalEncoder, MinMaxScaler, MaxAbsScaler)

from src.utility.nn.lineal import LinealNN

class CrossvalidationDataset(Dataset) :
    """
    Clase abstracta que hereda de dataset de pytorch y debe implementar
    metodos para poder realizar validacion cruzada con scikit learn
    """
    labels : dict[str,Any]
    class_targets : DataFrame
    regression_targets : DataFrame
    features : DataFrame
    @abstractmethod
    def split(self, crossvalidator : BaseCrossValidator) -> Generator:
        """
        Metodo abstracto que realiza la separacion de acuerdo a un validador cruzado
        """
    @abstractmethod
    def label_decode(self, label : str, pred : ndarray) -> ndarray :
        """
        Metodo que traduce un dataframe a tensor
        """
    @abstractmethod
    def encode(self, target : DataFrame) -> Tensor :
        """
        Metodo que traduce un dataframe a tensor
        """
    @abstractmethod
    def decode(self, pred : Tensor) -> DataFrame:
        """
        Metodo que traduce un tensor a dataframe
        """
    @abstractmethod
    def prediction(self, model : Module, dataloader : DataLoader | None = None) -> dict:
        """
        Metodo que devuelve las metricas del dataset o dataloader designado
        """
    @abstractmethod
    def loss_fn(self, pred : Tensor, target : Tensor) -> Tensor :
        """
        Metodo que implementa la funcion de perdida para el dataset
        """
    @abstractmethod
    def data_augmentation(self, dataframe : DataFrame, data_augment : int = 0) -> DataFrame :
        """
        A
        """

class CrossvalidationTensorDataset(CrossvalidationDataset) :
    """
    Implementacion de un dataset con validacion cruzada de scikit learn usando tensores de pytorch
    """
    tensors : dict[str,Tensor]
    dataset : TensorDataset
    binarized : dict[str,str,ArrayLike | spmatrix]
    encoded : dict[str,ArrayLike | spmatrix]
    labels : dict[str,Index]
    binarizers : dict[str,LabelBinarizer]
    encoders : dict[str,LabelEncoder]
    std_scalers : dict[str,StandardScaler]
    std_scaled : dict[str,ArrayLike | spmatrix]
    mm_scalers : dict[str,MinMaxScaler]
    mm_scaled : dict[str,ArrayLike | spmatrix]
    ma_scalers : dict[str,MaxAbsScaler]
    max_scaled : dict[str,ArrayLike | spmatrix]
    dataframe : DataFrame
    train : bool
    targets_tensor : Tensor
    features_tensor : Tensor
    targets_size : dict[str,int]
    features_size : dict[str,int]

    def __init__(
        self, dataframe : DataFrame,
        labels : dict[str,Index], data_augment : int = 0
    ) :
        dataframe = self.data_augmentation(dataframe,data_augment)
        self.class_weights = {}
        self.tensors = {}
        self.binarizers = {}
        self.binarized = {}
        self.encoders = {}
        self.encoded = {}
        self.std_scalers = {}
        self.std_scaled = {}
        self.mm_scalers = {}
        self.mm_scaled = {}
        self.ma_scalers = {}
        self.ma_scaled = {}
        self.dataframe = dataframe
        self.labels = labels
        self.labels['features'] = Index([
            *self.labels['ordinal features'], *self.labels['categorical features'],
            *self.labels['sparsed features'], *self.labels['offset features'],
            *self.labels['normal features'], *self.labels['regular features']
        ])
        self.labels['targets'] = Index([
            *self.labels['class targets'], *self.labels['regression targets']
        ])
        self.generate_features_tensors()
        self.generate_targets_tensors()
        self.generate_dataset()
        self.generate_tensors()

    def generate_features_tensors(self) :
        """
        A
        """
        for label in self.labels['sparsed features'] :
            self.ma_scalers[label] = MaxAbsScaler()
            self.ma_scaled[label] = self.ma_scalers[label].fit_transform(
                self.dataframe.loc[:,label].to_numpy().reshape(len(self.dataframe),-1)
            )
            self.tensors[label] = torch.tensor(
                data = self.ma_scaled[label],
                dtype = torch.double
            )
            self.tensors[label] = self.tensors[label].view(
                self.tensors[label].size(dim = 0), -1
            )
        for label in self.labels['offset features'] :
            self.mm_scalers[label] = MinMaxScaler()
            self.mm_scaled[label] = self.mm_scalers[label].fit_transform(
                self.dataframe.loc[:,label].to_numpy().reshape(len(self.dataframe),-1)
            )
            self.tensors[label] = torch.tensor(
                data = self.mm_scaled[label],
                dtype = torch.double
            )
            self.tensors[label] = self.tensors[label].view(
                self.tensors[label].size(dim = 0), -1
            )
        for label in self.labels['normal features'] :
            self.std_scalers[label] = StandardScaler()
            self.std_scaled[label] = self.std_scalers[label].fit_transform(
                self.dataframe.loc[:,label].to_numpy().reshape(len(self.dataframe),-1)
            )
            self.tensors[label] = torch.tensor(
                data = self.std_scaled[label],
                dtype = torch.double
            )
            self.tensors[label] = self.tensors[label].view(
                self.tensors[label].size(dim = 0), -1
            )
        for label in self.labels['categorical features'] :
            self.binarizers[label] = LabelBinarizer()
            self.binarized[label] = self.binarizers[label].fit_transform(
                self.dataframe.loc[:,label].to_numpy().ravel()
            )
            self.encoders[label] = LabelEncoder()
            self.encoded[label] = self.encoders[label].fit_transform(
                self.dataframe.loc[:,label].to_numpy().ravel()
            )
            self.tensors[label] = torch.tensor(
                data = self.binarized[label],
                dtype = torch.double
            )
            self.tensors[label] = self.tensors[label].view(
                self.tensors[label].size(dim = 0), -1
            )
        for label in self.labels['ordinal features'] :
            self.encoders[label] = OrdinalEncoder()
            self.encoded[label] = self.encoders[label].fit_transform(
                self.dataframe.loc[:,label].to_numpy().reshape(len(self.dataframe),-1)
            )
            self.tensors[label] = torch.tensor(
                data = self.encoded[label],
                dtype = torch.double
            )
            self.tensors[label] = self.tensors[label].view(
                self.tensors[label].size(dim = 0), -1
            )
        for label in self.labels['regular features'] :
            self.tensors[label] = torch.tensor(
                data = self.dataframe.loc[:,label].to_numpy(),
                dtype = torch.double
            )
            self.tensors[label] = self.tensors[label].view(
                self.tensors[label].size(dim = 0), -1
            )

    def generate_targets_tensors(self) :
        """
        A
        """
        for label in self.labels['class targets'] :
            self.binarizers[label] = LabelBinarizer()
            self.binarized[label] = self.binarizers[label].fit_transform(
                self.dataframe.loc[:,label].to_numpy().ravel()
            )
            self.encoders[label] = LabelEncoder()
            self.encoded[label] = self.encoders[label].fit_transform(
                self.dataframe.loc[:,label].to_numpy().ravel()
            )
            self.tensors[label] = torch.tensor(
                data = self.binarized[label],
                dtype = torch.double
            )
            self.class_weights[label] = (
                1 - self.tensors[label].mean(dim = 0)
            ) * (
                len(self.encoders[label].classes_) / (
                    len(self.encoders[label].classes_) - 1
                )
            )
            self.tensors[label] = self.tensors[label].view(
                self.tensors[label].size(dim = 0), -1
            )
        for label in self.labels['regression targets'] :
            self.tensors[label] = torch.tensor(
                data = self.dataframe.loc[:,label].to_numpy().reshape(len(self.dataframe),-1),
                dtype = torch.double
            )
            self.tensors[label] = self.tensors[label].view(
                self.tensors[label].size(dim = 0), -1
            )

    def generate_dataset(self) :
        """
        A
        """
        self.targets_tensor = torch.concat([
            self.tensors[label]
            for label in self.labels['targets']
        ], dim = 1)
        self.targets_size = self.targets_tensor.size(dim = 1)
        self.features_tensor = torch.concat([
            self.tensors[label]
            for label in self.labels['features']
        ], dim = 1)
        self.features_size = self.features_tensor.size(dim = 1)
        self.dataset = TensorDataset(self.features_tensor, self.targets_tensor)

    def generate_tensors(self) :
        """
        A
        """
        self.features = torch.concat(
            [
                self.tensors[label]
                for label in self.labels['features']
            ],
            dim = 1
        ).cpu().detach().numpy()
        if len(self.labels['regression targets']) > 0 :
            self.regression_targets = torch.concat(
                [
                    self.tensors[label]
                    for label in self.labels['regression targets']
                ],
                dim = 1
            ).cpu().detach().numpy()
        else :
            self.regression_targets = torch.empty((0,))
        if len(self.labels['class targets']) > 0 :
            self.class_targets = torch.concat(
                [
                    self.tensors[label]
                    for label in self.labels['class targets']
                ],
                dim = 1
            ).cpu().detach().numpy()
        else :
            self.class_targets = torch.empty((0,))

    def __len__(self) -> int :
        return len(self.dataset)

    def __getitem__(self, idx) -> tuple[Tensor, Tensor] :
        return self.dataset[idx]

    def encode(self, target : DataFrame)  -> Tensor:
        """
        Codifica los tensores en dataframes
        """
        return torch.column_stack([
            *[torch.tensor(
                self.binarizers['targets'][label].transform(
                    target[label].to_numpy()
                ), dtype = torch.double)
                for label in self.labels['class targets']
            ],
            torch.tensor(
                target[self.labels['regression targets']].to_numpy(),
                dtype = torch.double
            )
        ])

    def split_pred_tensor(self, pred : Tensor) :
        """
        Parte un tensor de acuerdo a las etiquetas de cada target
        """
        return dict(zip(
            self.labels['targets'],
            pred.split_with_sizes(
                [
                    self.tensors[label].size(dim = 1)
                    for label in self.labels['targets']
                ],
                dim = 1
            )
        ))

    def split_feat_tensor(self, features : Tensor) :
        """
        Parte un tensor de acuerdo a las etiquetas de cada caracteristica
        """
        return dict(zip(
            self.labels['targets'],
            features.split_with_sizes(
                [
                    self.tensors[label].size(dim = 1)
                    for label in self.labels['features']
                ],
                dim = 1
            )
        ))

    def label_decode(self, label : str, pred : ndarray) -> ndarray :
        """
        A
        """
        if label in self.labels['class targets'] and label in self.binarizers :
            return self.binarizers[label].inverse_transform(
                pred, threshold = 0
            ).squeeze()
        if label in self.labels['ordinal features'] and label in self.encoders :
            return self.encoders[label].inverse_transform(pred).squeeze()
        if label in self.labels['class features'] and label in self.binarizers :
            return self.binarizers[label].inverse_transform(pred).squeeze()
        if label in self.labels['normal features'] and label in self.std_scalers :
            return self.std_scalers[label].inverse_transform(pred).squeeze()
        if label in self.labels['offset features'] and label in self.mm_scalers :
            return self.mm_scalers[label].inverse_transform(pred).squeeze()
        if label in self.labels['sparsed features'] and label in self.ma_scalers :
            return self.ma_scalers[label].inverse_transform(pred).squeeze()
        return pred.squeeze()

    def decode(self, pred : Tensor)  -> DataFrame:
        """
        Decodifica tensores a dataframes
        """
        return DataFrame({ label : self.label_decode(label,data)
                    for label_list, tensor_list in [ tuple(
                        zip( *self.split_pred_tensor(pred).items() )
                    ) ]
                    for label, data in zip(
                        label_list,
                        ( tensor.cpu().detach().numpy() for tensor in tensor_list )
                    )
                })

    def split(self, crossvalidator : BaseCrossValidator) :
        """
        Implementacion de la particion de los conjuntos de entrenamiento y prueba en
        la validacion cruzada
        """
        yield from ((
                    Subset(self,train_index.tolist()),
                    Subset(self,test_index.tolist())
                    ) for train_index, test_index in crossvalidator.split(
                            X = self.dataframe[self.labels['features']].to_numpy(),
                            y = self.dataframe[self.labels['class targets'][0]].to_numpy().ravel()
                        )
                    )

    def prediction(self, model : Module, dataloader : DataLoader | None = None) :
        """
        A
        """
        with torch.inference_mode() :
            if dataloader is None :
                dataloader = DataLoader(dataset = self, batch_size = len(self))
            targets, predictions = (
                torch.vstack(tensors)
                for tensors in zip(*(
                    (y, model(X)) for X,y in dataloader
                ))
            )
        targets, predictions = (
            self.split_pred_tensor(tensor)
            for tensor in (targets, predictions)
        )
        results = {}
        for label in self.labels['regression targets'] :
            results[label] = (targets[label].numpy(), predictions[label].numpy())
        for label in self.labels['class targets'] :
            results[label] = (
                targets[label].numpy(),
                predictions[label].numpy(),
                self.label_decode(label,targets[label].numpy()),
                self.label_decode(label,predictions[label].numpy())
            )
        return results

    def loss_fn(self, pred : Tensor, target : Tensor) -> Tensor :
        """
        Function de perdida general para los tensores, considerando casos de clases y regresion
        """
        target_splitted = self.split_pred_tensor(target)
        pred_splitted = self.split_pred_tensor(pred)
        loss = torch.tensor(0, dtype = torch.double)
        zip_list = [
            *zip(pred_splitted.items(),target_splitted.values())
        ]
        for (label,pred),target in zip_list :
            if label in self.labels['regression targets'] :
                loss += mse_loss(pred, target, reduction = 'mean')
            elif pred.size(dim = 1) > 1 :
                loss += cross_entropy(
                    pred, target, reduction = 'mean',
                    weight = self.class_weights[label]
                )
            else :
                loss += binary_cross_entropy_with_logits(
                    pred, target, reduction = 'mean',
                    pos_weight = self.class_weights[label]
                )
        return loss / len(zip_list)

    def data_augmentation(self, dataframe : DataFrame, data_augment : int = 0) -> DataFrame:
        """
        A
        """
        return dataframe

def crossvalidate(dataset : CrossvalidationDataset, optimizer : tuple[type[Optimizer],dict],
                    loss_fn : Callable[[Tensor,Tensor],Tensor], arch : Iterable[int],
                    epochs : int, iterations : int, crossvalidator : BaseCrossValidator,
                    train_batches : int = 10,
                    extra_params : dict[str,Any] | None = None) :
    """
    Realiza la validacion cruzada sobre un conjunto, y un validador cruzado de scikit learn
    utilizando los hyperparametros que definen la arquitectura de la red neuronal
    """
    if extra_params is None :
        extra_params = {}
    base_optimizer,optimizer_kwargs = optimizer
    verbose = extra_params['verbose'] if 'verbose' in extra_params else False
    label_list = [
        'loss', 'train time', 'train loss',
        'scheduler','model','optimizer','test dataset',
        'train dataset','test size','train size',
        'train dataloader', 'test dataloader',
        'dataset'
    ]
    results = {label : [] for label in label_list}
    try :
        for it, (train_dataset, test_dataset) in zip(range(iterations),
            dataset.split(crossvalidator=crossvalidator)) :
            new_extra_params = extra_params.copy()
            if verbose :
                print(f'iteracion {it}')
            test_size = len(test_dataset)
            train_size = len(train_dataset)
            batch_size = train_size // train_batches
            train_dataloader = DataLoader(dataset = train_dataset, shuffle = True,
                                batch_size = batch_size,
                                drop_last = train_size % train_batches == 1)
            test_dataloader = DataLoader(
                dataset = test_dataset, batch_size = test_size
            )
            new_extra_params['test dataloader'] = test_dataloader
            model = LinealNN(C = arch, hyperparams = extra_params)
            optimizer = base_optimizer(model.parameters(), **optimizer_kwargs)
            if 'scheduler' in extra_params :
                base_scheduler,scheduler_kwargs = extra_params['scheduler']
                new_extra_params['scheduler'] = base_scheduler(optimizer, **scheduler_kwargs)
            ns_i = perf_counter_ns()
            train_loss = model.train_loop(dataloader = train_dataloader, epochs = epochs,
                                    optimizer = optimizer, loss_fn = loss_fn,
                                    extra_params = new_extra_params)
            ns_t = perf_counter_ns()
            model.load_state_dict(train_loss['model dict'])
            results['dataset'] += [dataset]
            results['test size'] += [test_size]
            results['train size'] += [train_size]
            results['optimizer'] += [optimizer]
            if 'scheduler' in new_extra_params :
                results['scheduler'] += [new_extra_params['scheduler']]
            results['train loss'] += [ train_loss ]
            results['model'] += [ model ]
            results['train time'] += [ns_t - ns_i]
            results['loss'] += [ train_loss['best'] ]
            results['test dataset'] += [test_dataset]
            results['test dataloader'] += [test_dataloader]
            results['train dataloader'] += [train_dataloader]
            results['train dataset'] += [train_dataset]
    except KeyboardInterrupt as kbi :
        print(f'{kbi}')
    array_sorted = numpy.argsort(array(results['loss']))
    for label in label_list :
        results[label] = [ results[label][it] for it in array_sorted ]
    return results
