"""
Modulo que contiene los metodos para realizar
validacion cruzada con scikit learn y pytorch
"""
from collections.abc import Iterable
import numpy
from numpy import ndarray
from numpy.typing import ArrayLike
from pandas import DataFrame, Index
import pandas
from scipy.sparse import spmatrix
import torch
from torch.nn import Module
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch.nn.functional import (
    cross_entropy, mse_loss,
    binary_cross_entropy_with_logits
)
from sklearn.model_selection import BaseCrossValidator
from sklearn.preprocessing import (
    LabelBinarizer, LabelEncoder, StandardScaler,
    OrdinalEncoder, MinMaxScaler, MaxAbsScaler
)
from src.utility.nn.cvdataset import CrossvalidationDataset


class CrossvalidationTensorDataset(CrossvalidationDataset):
    """
    Implementacion de un dataset con validacion cruzada de
    scikit learn usando tensores de pytorch
    """
    tensors: dict[str, Tensor]
    dataset: TensorDataset
    binarized: dict[str, ArrayLike | spmatrix]
    encoded: dict[str, ArrayLike | spmatrix]
    labels: dict[str, Index]
    binarizers: dict[str, LabelBinarizer]
    encoders: dict[str, LabelEncoder | OrdinalEncoder]
    std_scalers: dict[str, StandardScaler]
    std_scaled: dict[str, ArrayLike | spmatrix]
    mm_scalers: dict[str, MinMaxScaler]
    mm_scaled: dict[str, ArrayLike | spmatrix]
    ma_scalers: dict[str, MaxAbsScaler]
    max_scaled: dict[str, ArrayLike | spmatrix]
    dataframe: DataFrame
    train: bool
    targets_tensor: Tensor
    features_tensor: Tensor
    targets_size: int
    features_size: int
    crossvalidation_mode: bool
    augment_size: int | None
    train_indexes: list[int] | None
    test_indexes: list[int] | None
    augment_tensor: Tensor | None

    def __init__(
        self, dataframe: DataFrame,
        labels: dict[str, Index],
        data_augment: int = 0
    ):
        self.train_dataframe = None
        self.test_dataframe = None
        self.train_size = None
        self.test_size = None
        self.augment_tensor = None
        self.crossvalidation_mode = True
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
            *self.labels['ordinal features'], *
            self.labels['categorical features'],
            *self.labels['sparsed features'], *self.labels['offset features'],
            *self.labels['normal features'], *self.labels['regular features']
        ])
        self.labels['targets'] = Index([
            *self.labels['class targets'], *self.labels['regression targets']
        ])
        self.labels['all'] = Index([
            *self.labels['targets'], *self.labels['features']
        ])
        self.generate_features_tensors()
        self.generate_targets_tensors()
        self.generate_dataset()
        self.generate_tensors()
        self.data_augment(data_augment)

    def generate_features_tensors(self):
        """
        A
        """
        for label in self.labels['sparsed features']:
            self.ma_scalers[label] = MaxAbsScaler()
            self.ma_scaled[label] = self.ma_scalers[label].fit_transform(
                self.dataframe.loc[:, label].to_numpy().reshape(
                    len(self.dataframe), -1)
            )
            self.tensors[label] = torch.tensor(
                data=self.ma_scaled[label],
                dtype=torch.double
            )
            self.tensors[label] = self.tensors[label].view(
                self.tensors[label].size(dim=0), -1
            )
        for label in self.labels['offset features']:
            self.mm_scalers[label] = MinMaxScaler()
            self.mm_scaled[label] = self.mm_scalers[label].fit_transform(
                self.dataframe.loc[:, label].to_numpy().reshape(
                    len(self.dataframe), -1)
            )
            self.tensors[label] = torch.tensor(
                data=self.mm_scaled[label],
                dtype=torch.double
            )
            self.tensors[label] = self.tensors[label].view(
                self.tensors[label].size(dim=0), -1
            )
        for label in self.labels['normal features']:
            self.std_scalers[label] = StandardScaler()
            self.std_scaled[label] = self.std_scalers[label].fit_transform(
                self.dataframe.loc[:, label].to_numpy().reshape(
                    len(self.dataframe), -1)
            )
            self.tensors[label] = torch.tensor(
                data=self.std_scaled[label],
                dtype=torch.double
            )
            self.tensors[label] = self.tensors[label].view(
                self.tensors[label].size(dim=0), -1
            )
        for label in self.labels['categorical features']:
            self.binarizers[label] = LabelBinarizer()
            self.binarized[label] = self.binarizers[label].fit_transform(
                self.dataframe.loc[:, label].to_numpy().ravel()
            )
            self.encoders[label] = LabelEncoder()
            self.encoded[label] = self.encoders[label].fit_transform(
                self.dataframe.loc[:, label].to_numpy().ravel()
            )
            self.tensors[label] = torch.tensor(
                data=self.binarized[label],
                dtype=torch.double
            )
            self.tensors[label] = self.tensors[label].view(
                self.tensors[label].size(dim=0), -1
            )
        for label in self.labels['ordinal features']:
            self.encoders[label] = OrdinalEncoder()
            self.encoded[label] = self.encoders[label].fit_transform(
                self.dataframe.loc[:, label].to_numpy().reshape(
                    len(self.dataframe), -1)
            )
            self.tensors[label] = torch.tensor(
                data=self.encoded[label],
                dtype=torch.double
            )
            self.tensors[label] = self.tensors[label].view(
                self.tensors[label].size(dim=0), -1
            )
        for label in self.labels['regular features']:
            self.tensors[label] = torch.tensor(
                data=self.dataframe.loc[:, label].to_numpy(),
                dtype=torch.double
            )
            self.tensors[label] = self.tensors[label].view(
                self.tensors[label].size(dim=0), -1
            )

    def generate_targets_tensors(self):
        """
        A
        """
        for label in self.labels['class targets']:
            self.binarizers[label] = LabelBinarizer()
            self.binarized[label] = self.binarizers[label].fit_transform(
                self.dataframe.loc[:, label].to_numpy().ravel()
            )
            self.encoders[label] = LabelEncoder()
            self.encoded[label] = self.encoders[label].fit_transform(
                self.dataframe.loc[:, label].to_numpy().ravel()
            )
            self.tensors[label] = torch.tensor(
                data=self.binarized[label],
                dtype=torch.double
            )
            self.class_weights[label] = (
                1 - self.tensors[label].mean(dim=0)
            ) * (
                len(self.encoders[label].classes_) / (  # type: ignore
                    len(self.encoders[label].classes_) - 1  # type: ignore
                )
            )
            self.tensors[label] = self.tensors[label].view(
                self.tensors[label].size(dim=0), -1
            )
        for label in self.labels['regression targets']:
            self.tensors[label] = torch.tensor(
                data=self.dataframe.loc[:, label].to_numpy().reshape(
                    len(self.dataframe), -1),
                dtype=torch.double
            )
            self.tensors[label] = self.tensors[label].view(
                self.tensors[label].size(dim=0), -1
            )

    def generate_dataset(self):
        """
        A
        """
        self.targets_tensor = torch.concat([
            self.tensors[label]
            for label in self.labels['targets']
        ], dim=1)
        self.targets_size = self.targets_tensor.size(dim=1)
        self.features_tensor = torch.concat([
            self.tensors[label]
            for label in self.labels['features']
        ], dim=1)
        self.features_size = self.features_tensor.size(dim=1)
        self.dataset = TensorDataset(self.features_tensor, self.targets_tensor)

    def generate_tensors(self):
        """
        A
        """
        self.features = DataFrame(
            torch.concat(
                [
                    self.tensors[label]
                    for label in self.labels['features']
                ],
                dim=1
            ).cpu().detach().numpy(),
            columns=self.labels['features']
        )
        if len(self.labels['regression targets']) > 0:
            self.regression_targets = DataFrame(
                torch.concat(
                    [
                        self.tensors[label]
                        for label in self.labels['regression targets']
                    ],
                    dim=1
                ).cpu().detach().numpy(),
                columns=self.labels['regression targets']
            )
        else:
            self.regression_targets = DataFrame()
        if len(self.labels['class targets']) > 0:
            self.class_targets = DataFrame(
                torch.concat(
                    [
                        self.tensors[label]
                        for label in self.labels['class targets']
                    ],
                    dim=1
                ).cpu().detach().numpy(),
                columns=self.labels['class targets']
            )
        else:
            self.class_targets = DataFrame()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def encode(self, target: DataFrame) -> Tensor:
        """
        Codifica los tensores en dataframes
        """
        return torch.column_stack([
            *[
                torch.tensor(
                    self.binarizers[label].transform(
                        target[label].to_numpy()
                    ),
                    dtype=torch.double
                )
                for label in self.labels['class targets']
            ],
            torch.tensor(
                target[self.labels['regression targets']].to_numpy(),
                dtype=torch.double
            )
        ])

    def split_pred_tensor(self, pred: Tensor):
        """
        Parte un tensor de acuerdo a las etiquetas de cada target
        """
        return dict(zip(
            self.labels['targets'],
            pred.split_with_sizes(
                [
                    self.tensors[label].size(dim=1)
                    for label in self.labels['targets']
                ],
                dim=1
            )
        ))

    def split_feat_tensor(self, features: Tensor):
        """
        Parte un tensor de acuerdo a las etiquetas de cada caracteristica
        """
        return dict(zip(
            self.labels['targets'],
            features.split_with_sizes(
                [
                    self.tensors[label].size(dim=1)
                    for label in self.labels['features']
                ],
                dim=1
            )
        ))

    def label_decode(self, label: str, pred: ndarray) -> ndarray:
        """
        A
        """
        if (
            label in self.labels['class targets']
            and label in self.binarizers
        ):
            return self.binarizers[label].inverse_transform(
                pred, threshold=0.5
            ).squeeze()  # type: ignore
        elif (
            label in self.labels['ordinal features']
            and label in self.encoders
        ):
            return self.encoders[label].inverse_transform(pred).squeeze()
        elif (
            label in self.labels['class features']
            and label in self.binarizers
        ):
            return self.binarizers[label].inverse_transform(
                pred
            ).squeeze()  # type: ignore
        elif (
            label in self.labels['normal features']
            and label in self.std_scalers
        ):
            return self.std_scalers[label].inverse_transform(pred).squeeze()
        elif (
            label in self.labels['offset features']
            and label in self.mm_scalers
        ):
            return self.mm_scalers[label].inverse_transform(pred).squeeze()
        elif (
            label in self.labels['sparsed features']
            and label in self.ma_scalers
        ):
            return self.ma_scalers[label].inverse_transform(pred).squeeze()
        return pred.squeeze()

    def decode(self, pred: Tensor) -> DataFrame:
        """
        Decodifica tensores a dataframes
        """
        return DataFrame({
            label: self.label_decode(label, data)
            for label_list, tensor_list in [tuple(
                zip(*self.split_pred_tensor(pred).items())
            )]
            for label, data in zip(
                label_list,
                (tensor.cpu().detach().numpy() for tensor in tensor_list)
            )
        })

    def split(self, crossvalidator: BaseCrossValidator):
        """
        Implementacion de la particion de los conjuntos de
        entrenamiento y prueba en la validacion cruzada
        """
        if (
            self.crossvalidation_mode
            or self.train_indexes is None
            or self.test_indexes is None
        ):
            yield from ((
                        Subset(self, train_index.tolist()),
                        Subset(self, test_index.tolist())
                        ) for train_index, test_index in crossvalidator.split(
                X=self.dataframe[self.labels['features']].to_numpy(),
                y=self.dataframe[
                    self.labels['class targets'][0]
                ].to_numpy().ravel()
            )
            )
        else:
            yield (
                Subset(self, numpy.random.permutation(
                    self.train_indexes
                ).tolist()),
                Subset(self, numpy.random.permutation(
                    self.test_indexes
                ).tolist())
            )

    def calculate_probabilities(self, pred: dict[str, Tensor]):
        """
        A
        """
        return {
            label: (
                tensor if label not in self.labels['class targets']
                else tensor.sigmoid() if tensor.size(dim=1) < 2
                else tensor.softmax(dim=1)
            )
            for label, tensor in pred.items()
        }

    def prediction(self, model: Module, dataloader: DataLoader | None = None):
        """
        A
        """
        with torch.inference_mode():
            if dataloader is None:
                dataloader = DataLoader(dataset=self, batch_size=len(self))
            targets, predictions = (
                torch.vstack(tensors)
                for tensors in zip(*(
                    (y, model(X)) for X, y in dataloader
                ))
            )
        targets = self.split_pred_tensor(targets)
        predictions = self.calculate_probabilities(
            self.split_pred_tensor(predictions)
        )
        results = {}
        for label in self.labels['regression targets']:
            results[label] = (
                targets[label].cpu().detach().numpy(),
                predictions[label].cpu().detach().numpy()
            )
        for label in self.labels['class targets']:
            results[label] = (
                targets[label].cpu().detach().numpy(),
                predictions[label].cpu().detach().numpy(),
                self.label_decode(
                    label, targets[label].cpu().detach().numpy()),
                self.label_decode(
                    label, predictions[label].cpu().detach().numpy())
            )
        return results

    def loss_fn(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Function de perdida general para los tensores, considerando
        casos de clases y regresion
        """
        target_splitted = self.split_pred_tensor(target)
        pred_splitted = self.split_pred_tensor(pred)
        loss = torch.tensor(0, dtype=torch.double)
        zip_list = [
            *zip(pred_splitted.items(), target_splitted.values())
        ]
        for (label, pred), target in zip_list:
            if label in self.labels['regression targets']:
                loss += mse_loss(pred, target, reduction='mean')
            elif pred.size(dim=1) > 1:
                loss += cross_entropy(
                    pred, target, reduction='mean',
                    weight=self.class_weights[label]
                )
            else:
                loss += binary_cross_entropy_with_logits(
                    pred, target, reduction='mean',
                    pos_weight=self.class_weights[label]
                )
        return loss / len(zip_list)

    @classmethod
    def from_dataframes(
        cls, labels: dict[str, Index],
        train: DataFrame, test: DataFrame,
        data_augment: int = 0
    ):
        dataframe = pandas.concat(
            (train, test),
            axis='index',
            ignore_index=True
        )
        dataset = cls(dataframe, labels, data_augment)
        dataset.set_indexes(range(len(train)), False)
        dataset.set_indexes(range(len(train), len(train) + len(test)), True)
        dataset.crossvalidation_mode = False

    def set_indexes(self, indexes: Iterable[int], is_test: bool = True):
        """
        A
        """
        if is_test:
            self.test_indexes = list(indexes)
        else:
            self.train_indexes = list(indexes)

    def to_dataframe(
        self, subset: str | Iterable[int] | None = None,
        label_type: str | Iterable[str] | None = None, raw: bool = False
    ):
        """
        A
        """
        dataframe = None
        if label_type is None:
            label_type = self.labels['all']
        elif isinstance(label_type, str):
            label_type = self.labels[label_type]
        else:
            label_type = list(label_type)
        data = {}
        for label in label_type:
            if subset is None:
                data[label] = self.tensors[label].cpu().detach().numpy()
            elif isinstance(subset, str):
                match subset:
                    case 'all':
                        data[label] = self.tensors[
                            label
                        ].cpu().detach().numpy()
                    case 'train':
                        if self.train_indexes is not None:
                            data[label] = self.tensors[label][
                                self.train_indexes
                            ].cpu().detach().numpy()
                        else:
                            data[label] = self.tensors[
                                label
                            ].cpu().detach().numpy()
                    case 'test':
                        if self.test_indexes is not None:
                            data[label] = self.tensors[label][
                                self.test_indexes
                            ].cpu().detach().numpy()
                        else:
                            data[label] = self.tensors[
                                label
                            ].cpu().detach().numpy()
                    case _:
                        return None
            else:
                data[label] = self.tensors[label][
                    list(subset)
                ].cpu().detach().numpy()
        if raw:
            dataframe_dict = {}
            for label, label_data in data.items():
                if len(label_data.shape) == 2:
                    for i in range(label_data.shape[1]):
                        dataframe_dict[f'{label}_{i}'] = label_data[
                            :, i
                        ].squeeze()
                else:
                    dataframe_dict[label] = label_data.squeeze()
            dataframe = DataFrame(dataframe_dict)
        else:
            dataframe_dict = {}
            for label, label_data in data.items():
                dataframe_dict[label] = self.label_decode(
                    label,
                    label_data
                ).squeeze()
            dataframe = DataFrame(dataframe_dict)
        return dataframe

    def data_augment(self, data_augment: int = 0):
        """
        A
        """
        self.augment_size = data_augment
