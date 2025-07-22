"""
Modulo que contiene los metodos para realizar
validacion cruzada con scikit learn y pytorch
"""
from collections.abc import Iterable
import numpy
import numpy.random as numpyrand
from numpy.random import SeedSequence, PCG64
from numpy import ndarray, uint64
from numpy.typing import ArrayLike
from pandas import DataFrame, Index
import pandas
from scipy.sparse import spmatrix
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch.nn.functional import (
    cross_entropy, mse_loss,
    binary_cross_entropy_with_logits
)
from sklearn.model_selection import BaseCrossValidator, LeaveOneOut
from sklearn.preprocessing import (
    LabelBinarizer, LabelEncoder, StandardScaler,
    OrdinalEncoder, MinMaxScaler, MaxAbsScaler
)
from src.utility.nn.lineal import LinealNN, set_defaults
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
    original_size: int
    augment_size: int | None
    size: int
    train_indices: list[int] | None
    test_indices: list[int] | None
    crossvalidator: BaseCrossValidator

    def __init__(
        self, dataframe: DataFrame,
        labels: dict[str, Index],
        crossvalidator: BaseCrossValidator,
        data_augment: int = 0,
        **kwargs
    ):
        set_defaults()
        self.crossvalidator = crossvalidator
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
        self.original_size = len(dataframe)
        self.labels = labels
        self.labels['features'] = Index([
            *self.labels['ordinal features'],
            *self.labels['categorical features'],
            *self.labels['sparsed features'], *self.labels['offset features'],
            *self.labels['normal features'], *self.labels['regular features']
        ])
        self.labels['targets'] = Index([
            *self.labels['class targets'], *self.labels['regression targets']
        ])
        self.labels['all'] = Index([
            *self.labels['targets'], *self.labels['features']
        ])
        self.__generate_features_tensors()
        self.__generate_targets_tensors()
        seed = None
        if 'seed' in kwargs:
            seed = kwargs['seed']
        self.data_augment(data_augment, seed)
        self.test_indices = None
        self.train_indices = None
        if (
            'test_indices' in kwargs
            and kwargs['test_indices'] is not None
        ):
            self.set_test_indexes(kwargs['test_indices'])
        if (
            'train_indices' in kwargs
            and kwargs['train_indices'] is not None
        ):
            self.set_train_indexes(kwargs['train_indices'])

    def generator_dict(self):
        """
        A
        """
        generator = {}
        generator['test_indices'] = self.test_indices
        generator['train_indices'] = self.train_indices
        generator['labels'] = self.labels
        generator['dataframe'] = self.dataframe
        generator['seed'] = self.seed
        generator['data_augment'] = self.augment_size
        generator['crossvalidator'] = self.crossvalidator
        return generator

    @classmethod
    def from_generator_dict(cls, generator: dict):
        """
        A
        """
        return cls(**generator)

    def __generate_features_tensors(self):
        set_defaults()
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

    def __generate_targets_tensors(self):
        set_defaults()
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

    def __generate_dataset(self):
        set_defaults()
        self.size = self.original_size + (
            self.augment_size
            if self.augment_size is not None
            else 0
        )
        self.targets_tensor = torch.concat(
            [
                torch.zeros((self.size, 0)),
                *(
                    self.tensors[label]
                    for label in self.labels['targets']
                )
            ],
            dim=1
        )
        self.targets_size = self.targets_tensor.size(dim=1)
        self.features_tensor = torch.concat(
            [
                torch.zeros((self.size, 0)),
                *(
                    self.tensors[label]
                    for label in self.labels['features']
                )
            ],
            dim=1
        )
        self.features_size = self.features_tensor.size(dim=1)
        self.dataset = TensorDataset(self.features_tensor, self.targets_tensor)

    def __generate_tensors(self):
        set_defaults()
        self.features = DataFrame(
            torch.concat(
                [
                    torch.zeros((self.size, 0)),
                    *(
                        self.tensors[label]
                        for label in self.labels['features']
                    )
                ],
                dim=1
            ).cpu().detach().numpy(),
            columns=self.labels['features']
        )
        if len(self.labels['regression targets']) > 0:
            self.regression_targets = DataFrame(
                torch.concat(
                    [
                        torch.zeros((self.size, 0)),
                        *(
                            self.tensors[label]
                            for label in self.labels['regression targets']
                        )
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
                        torch.zeros((self.size, 0)),
                        *(
                            self.tensors[label]
                            for label in self.labels['class targets']
                        )
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
        set_defaults()
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
        set_defaults()
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
        set_defaults()
        return dict(zip(
            self.labels['features'],
            features.split_with_sizes(
                [
                    self.tensors[label].size(dim=1)
                    for label in self.labels['features']
                ],
                dim=1
            )
        ))

    def split_feat_array(self, features: ndarray):
        """
        Parte un tensor de acuerdo a las etiquetas de cada caracteristica
        """
        set_defaults()
        return dict(zip(
            self.labels['features'],
            numpy.array_split(
                features,
                numpy.array([
                    self.tensors[label].size(dim=1)
                    for label in self.labels['features']
                ]).cumsum(),
                axis=1
            )
        ))

    def split_pred_array(self, pred: ndarray):
        """
        Parte un tensor de acuerdo a las etiquetas de cada caracteristica
        """
        set_defaults()
        return dict(zip(
            self.labels['targets'],
            numpy.array_split(
                pred,
                numpy.array([
                    self.tensors[label].size(dim=1)
                    for label in self.labels['targets']
                ]).cumsum(),
                axis=1
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

    def decode(self, pred: ndarray) -> DataFrame:
        """
        Decodifica tensores a dataframes
        """
        return DataFrame({
            label: self.label_decode(label, data)
            for label_list, array_list in [(
                *zip(*self.split_pred_array(pred).items()),
            )]
            for label, data in zip(label_list, array_list)
        })

    def split(self):
        """
        Implementacion de la particion de los conjuntos de
        entrenamiento y prueba en la validacion cruzada
        """
        set_defaults()
        if (
            self.crossvalidation_mode
            or self.train_indices is None
            or self.test_indices is None
        ):
            yield from (
                (
                    Subset(self, train_index.tolist()),
                    Subset(self, test_index.tolist())
                )
                for train_index, test_index in self.crossvalidator.split(
                    X=self.dataframe[self.labels['features']].to_numpy(),
                    y=self.dataframe[
                        self.labels['class targets'][0]
                    ].to_numpy().ravel()
                )
            )
        else:
            yield (
                Subset(self, numpy.random.permutation(
                    self.train_indices
                ).tolist()),
                Subset(self, numpy.random.permutation(
                    self.test_indices
                ).tolist())
            )

    def prediction(self, model: LinealNN, indices: list[int] | None = None):
        """
        A
        """
        set_defaults()
        with torch.inference_mode():
            if indices is not None:
                dataset = Subset(self, indices)
            else:
                dataset = self
            dataloader = DataLoader(dataset=dataset, batch_size=len(dataset))
            targets_arr, predictions_arr, = zip(*(
                (prediction, target)
                for prediction, target in (
                    (model.inference(X), y.cpu().detach().numpy())
                    for X, y in dataloader
                )
            ))
        targets = self.split_pred_array(numpy.vstack(targets_arr))
        predictions = self.split_pred_array(numpy.vstack(predictions_arr))
        results = {}
        for label in self.labels['regression targets']:
            results[label] = (
                targets[label],
                predictions[label]
            )
        for label in self.labels['class targets']:
            results[label] = (
                targets[label],
                predictions[label],
                self.label_decode(label, targets[label]),
                self.label_decode(label, predictions[label])
            )
        return results

    @classmethod
    def from_dataframes(
        cls, labels: dict[str, Index],
        train: DataFrame, test: DataFrame,
        crossvalidator: BaseCrossValidator | None = None,
        data_augment: int = 0
    ):
        dataframe = pandas.concat(
            (train, test),
            axis='index',
            ignore_index=True
        )
        if crossvalidator is None:
            crossvalidator = LeaveOneOut()
        dataset = cls(dataframe, labels, crossvalidator, data_augment)
        dataset.set_train_indexes(range(len(train)))
        dataset.crossvalidation_mode = False

    def set_train_indexes(self, indexes: Iterable[int]):
        """
        A
        """
        self.train_indices = list(indexes)
        self.test_indices = [
            idx for idx in range(len(self))
            if idx not in self.train_indices
        ]

    def set_test_indexes(self, indexes: Iterable[int]):
        """
        A
        """
        self.test_indices = list(indexes)
        self.train_indices = [
            idx for idx in range(len(self))
            if idx not in self.test_indices
        ]

    def to_dataframe(
        self, subset: str | Iterable[int] | None = None,
        label_type: str | Iterable[str] | None = None, raw: bool = False
    ):
        """
        A
        """
        set_defaults()
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
                        if self.train_indices is not None:
                            data[label] = self.tensors[label][
                                self.train_indices
                            ].cpu().detach().numpy()
                        else:
                            data[label] = self.tensors[
                                label
                            ].cpu().detach().numpy()
                    case 'test':
                        if self.test_indices is not None:
                            data[label] = self.tensors[label][
                                self.test_indices
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

    def data_augment(self, data_augment: int = 0, seed: int | None = None):
        """
        A
        """
        set_defaults()
        ss = SeedSequence()
        if seed is not None:
            ss = SeedSequence(entropy=seed)
        self.seed = ss.entropy
        num_gen = numpyrand.default_rng(seed=PCG64(seed=ss))
        tor_gen = torch.Generator(device=torch.get_default_device())
        tor_gen.manual_seed(
            int(num_gen.integers(
                low=0, high=0xffff_ffff_ffff_ffff,
                endpoint=True, dtype=uint64
            ))
        )
        self.augment_size = data_augment
        added_tensor = {
            label: []
            for label in (*self.labels['features'], * self.labels['targets'])
        }
        for sample in num_gen.integers(
            low=0, high=self.original_size, size=data_augment
        ).tolist():
            for label in self.labels['ordinal features']:
                tensor = self.tensors[label][sample, :]
                if tensor.dim() < 2:
                    tensor = tensor.unsqueeze(dim=1)
                added_tensor[label] += [(
                    tensor - 0.5
                    + torch.rand(
                        tensor.size(),
                        layout=tensor.layout,
                        dtype=tensor.dtype,
                        device=tensor.device,
                        generator=tor_gen
                    )
                )]
            for label in self.labels['categorical features']:
                if tensor.size(dim=1) == 1:
                    tensor = self.tensors[label][sample, :]
                    if tensor.dim() < 2:
                        tensor = tensor.unsqueeze(dim=1)
                    added_tensor[label] += [(
                        0.5 * torch.rand(
                            tensor.size(),
                            layout=tensor.layout,
                            dtype=tensor.dtype,
                            device=tensor.device,
                            generator=tor_gen
                        )
                        + 0.5 * (tensor > 0.5).double()
                    )]
                else:
                    tensor = self.tensors[label][sample, :]
                    if tensor.dim() < 2:
                        tensor = tensor.unsqueeze(dim=1)
                    added_tensor[label] += [((
                        tensor + torch.randn(
                            tensor.size(),
                            layout=tensor.layout,
                            dtype=tensor.dtype,
                            device=tensor.device,
                            generator=tor_gen
                        ).softmax(dim=1)
                    ).softmax(dim=1))]
            for label in self.labels['normal features']:
                tensor = self.tensors[label][sample, :]
                if tensor.dim() < 2:
                    tensor = tensor.unsqueeze(dim=1)
                added_tensor[label] += [(
                    tensor + torch.rand(
                            tensor.size(),
                            layout=tensor.layout,
                            dtype=tensor.dtype,
                            device=tensor.device,
                            generator=tor_gen
                    ) / 10 * torch.tensor(self.std_scalers[label].var_)
                )]
            for label in self.labels['regular features']:
                tensor = self.tensors[label][sample, :]
                if tensor.dim() < 2:
                    tensor = tensor.unsqueeze(dim=1)
                added_tensor[label] += [tensor]
            for label in self.labels['sparsed features']:
                tensor = self.tensors[label][sample, :]
                if tensor.dim() < 2:
                    tensor = tensor.unsqueeze(dim=1)
                added_tensor[label] += [(
                    tensor + (tensor == 0).double()
                    * torch.rand(
                        tensor.size(),
                        layout=tensor.layout,
                        dtype=tensor.dtype,
                        device=tensor.device,
                        generator=tor_gen
                    ) / 100 * torch.tensor(self.ma_scalers[label].scale_)
                )]
            for label in self.labels['offset features']:
                tensor = self.tensors[label][sample, :]
                if tensor.dim() < 2:
                    tensor = tensor.unsqueeze(dim=1)
                added_tensor[label] += [(
                    tensor + torch.rand(
                        tensor.size(),
                        layout=tensor.layout,
                        dtype=tensor.dtype,
                        device=tensor.device,
                        generator=tor_gen
                    ) / 100 * torch.tensor(self.mm_scalers[label].scale_)
                )]
            for label in self.labels['regression targets']:
                tensor = self.tensors[label][sample, :]
                if tensor.dim() < 2:
                    tensor = tensor.unsqueeze(dim=1)
                added_tensor[label] += [tensor]
            for label in self.labels['class targets']:
                tensor = self.tensors[label][sample, :]
                if tensor.dim() < 2:
                    tensor = tensor.unsqueeze(dim=1)
                if tensor.size(dim=1) == 1:
                    added_tensor[label] += [(
                        0.5 * torch.rand(
                            tensor.size(),
                            layout=tensor.layout,
                            dtype=tensor.dtype,
                            device=tensor.device,
                            generator=tor_gen
                        ) + 0.5 * (tensor > 0.5).double()
                    )]
                else:
                    added_tensor[label] += [((
                        tensor + torch.randn(
                            tensor.size(),
                            layout=tensor.layout,
                            dtype=tensor.dtype,
                            device=tensor.device,
                            generator=tor_gen
                        ).softmax(dim=1)
                    ).softmax(dim=1))]
        for label in (*self.labels['features'], * self.labels['targets']):
            tensor = self.tensors[label][:self.original_size, :]
            if tensor.dim() < 2:
                tensor = tensor.unsqueeze(dim=1)
            self.tensors[label] = torch.concat(
                (
                    tensor,
                    *added_tensor[label]
                ),
                dim=0
            )
        self.__generate_dataset()
        self.__generate_tensors()

    def inference_function(self, pred: Tensor) -> Tensor:
        """
        Function de perdida general para los tensores, considerando
        casos de clases y regresion
        """
        set_defaults()
        return torch.concat(
            [
                pred
                if label in self.labels['regression targets']
                else pred.sigmoid()
                if pred.size(dim=1) == 1
                else pred.softmax(dim=1)
                for label, pred in self.split_pred_tensor(pred).items()
            ],
            dim=1
        )

    def loss_fn(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Function de perdida general para los tensores, considerando
        casos de clases y regresion
        """
        set_defaults()
        target_splitted = self.split_pred_tensor(target)
        pred_splitted = self.split_pred_tensor(pred)
        loss = torch.tensor(0.0, dtype=torch.double)
        for label in self.labels['regression targets']:
            target = target_splitted[label]
            pred = pred_splitted[label]
            loss += mse_loss(pred, target, reduction='mean')
        for label in self.labels['class targets']:
            target = target_splitted[label]
            pred = pred_splitted[label]
            if pred.size(dim=1) > 1:
                loss += cross_entropy(
                    pred, target, reduction='mean',
                    weight=self.class_weights[label]
                )
            else:
                loss += binary_cross_entropy_with_logits(
                    pred, target, reduction='mean',
                    pos_weight=self.class_weights[label]
                )
        return loss / len(self.labels['targets'])
