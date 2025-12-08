"""
A
"""
from itertools import accumulate
import numpy
import numpy.random as numpyrand
from numpy import ndarray
import pandas
from pandas import DataFrame
from sklearn.model_selection import BaseCrossValidator, train_test_split
from sklearn.preprocessing import (
    MaxAbsScaler, MinMaxScaler, StandardScaler, LabelBinarizer, OrdinalEncoder
)
import torch
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss, cross_entropy
from torch import Tensor
from utility.nn import torchdefault


def noise_tensor_augment(
    tensor_dict: dict[str, ndarray],
    labels: dict,
    encoder: dict,
    data_augment: int = 0,
    scaler: float = 1000,
    seed: int | None = None
) -> dict[str, ndarray]:
    """
    A
    """
    generator = numpyrand.default_rng(seed)
    augment_samples = generator.integers(0, len(tensor_dict['pandas index']), data_augment)
    augmented_dict = {label: [] for label in tensor_dict}
    for i, sample in enumerate(augment_samples):
        augmented_dict['pandas index'] += [f'aug_{i}_{tensor_dict["pandas index"][sample, :]}']
        for label in labels['ordinal features']:
            tensor_array = tensor_dict[label][sample, :]
            augmented_dict[label] += [tensor_array - 0.5 + generator.random(tensor_array.shape)]
        for label in labels['categorical features']:
            tensor_array = tensor_dict[label][sample, :]
            augmented_dict[label] += [tensor_array - 0.5 + generator.random(tensor_array.shape)]
        for label in labels['normal features']:
            tensor_array = tensor_dict[label][sample, :]
            normal_scale = encoder[label].scale_ / scaler
            augmented_dict[label] += [tensor_array + generator.normal(0.0, normal_scale, tensor_array.shape)]
        for label in labels['sparsed features']:
            tensor_array = tensor_dict[label][sample, :]
            sparsed_scale = encoder[label].scale_ * scaler
            augmented_dict[label] += [tensor_array + generator.random(tensor_array.shape) / sparsed_scale]
        for label in labels['offset features']:
            tensor_array = tensor_dict[label][sample, :]
            offset_scale = encoder[label].scale_ * scaler
            augmented_dict[label] += [tensor_array + generator.random(tensor_array.shape) / offset_scale]
        for label in labels['class targets']:
            tensor_array = tensor_dict[label][sample, :]
            alpha = generator.random()
            num_classes = encoder[label].classes_
            augmented_dict[label] += [(1 - alpha) * tensor_array + alpha / num_classes]
        for label in labels['regression targets']:
            tensor_array = tensor_dict[label][sample, :]
            normal_scale = 1.0 / scaler
            augmented_dict[label] += [tensor_array + generator.normal(0.0, normal_scale, tensor_array.shape)]
        for label in labels['regular features']:
            tensor_array = tensor_dict[label][sample, :]
            normal_scale = 1.0 / scaler
            augmented_dict[label] += [tensor_array + generator.normal(0.0, normal_scale, tensor_array.shape)]
    new_tensor_dict = {}
    for label in tensor_dict:
        if data_augment > 0:
            new_tensor_dict[label] = numpy.concat([
                    tensor_dict[label],
                    numpy.concat(augmented_dict[label], axis=0)
                ],
                axis=0
            )
        else:
            new_tensor_dict[label] = tensor_dict[label]
    return new_tensor_dict

class CsvDataset:
    """
    A
    """
    original_dataframe: DataFrame
    train_dataframe: DataFrame
    crossvalidation_dataframe: DataFrame
    test_dataframe: DataFrame
    validation_dataframe: DataFrame
    augmented_dataframe: DataFrame | None
    encoder: dict
    data_augment: int
    scaler: float
    seed: int

    def state_dict(self):
        ds_state_dict = {}
        ds_state_dict['train_dataframe'] = self.train_dataframe.to_dict()
        ds_state_dict['test_dataframe'] = self.test_dataframe.to_dict()
        ds_state_dict['validation_dataframe'] = self.validation_dataframe.to_dict()
        if self.augmented_dataframe is not None:
            ds_state_dict['augmented_dataframe'] = self.augmented_dataframe.to_dict()
        else:
            ds_state_dict['augmented_dataframe'] = None
        ds_state_dict['labels'] = self.labels
        ds_state_dict['seed'] = self.seed
        ds_state_dict['scaler'] = self.scaler
        ds_state_dict['data_augment'] = self.data_augment
        return ds_state_dict
    
    @classmethod
    def from_state_dict(cls, state_dict: dict):
        return cls(
            train_dataframe=DataFrame(state_dict['train_dataframe']),
            test_dataframe=DataFrame(state_dict['test_dataframe']),
            validation_dataframe=DataFrame(state_dict['validation_dataframe']),
            labels=state_dict['labels'],
            augmented_dataframe=DataFrame(state_dict['augmented_dataframe']) if state_dict['augmented_dataframe'] is not None else None,
            seed=state_dict['seed'],
            scaler=state_dict['scaler'],
            data_augment=state_dict['data_augment']
        )

    @classmethod
    def from_dataframe(
        cls,      dataframe: DataFrame,         labels: dict[str, list[str]],
        train : int | float | None = None,      test : int | float | None = None,
        validation : int | float | None = None, seed: int | None = None, **kwargs
    ):
        ss = numpyrand.SeedSequence(seed).spawn(2)
        generator = numpyrand.default_rng(ss[0])
        if isinstance(train, float) and isinstance(test, float) and isinstance(validation, float):
            assert 0.0 <= train and 0.0 <= test and 0.0 <= validation and train + test + validation <= 1.0
        remain_dataframe = dataframe
        train_dataframe, remain_dataframe = train_test_split(
            remain_dataframe,
            train_size=train,
            random_state=generator.integers(0, 2 ** 32 - 1),
            shuffle=True,
            stratify=remain_dataframe.loc[:, labels['class targets']]
        )
        test_dataframe, remain_dataframe = train_test_split(
            remain_dataframe,
            train_size=test,
            random_state=generator.integers(0, 2 ** 32 - 1),
            shuffle=True,
            stratify=remain_dataframe.loc[:, labels['class targets']]
        )
        validation_dataframe, remain_dataframe = train_test_split(
            remain_dataframe,
            train_size=validation,
            random_state=generator.integers(0, 2 ** 32 - 1),
            shuffle=True,
            stratify=remain_dataframe.loc[:, labels['class targets']]
        )
        dataset = cls(
            train_dataframe=train_dataframe,
            test_dataframe=test_dataframe,
            validation_dataframe=validation_dataframe,
            labels=labels,
            seed=ss[1].entropy, # type: ignore
            **kwargs
        )
        return dataset

    def __init__(
        self,                         train_dataframe: DataFrame,
        test_dataframe: DataFrame,    validation_dataframe: DataFrame,
        labels: dict[str, list[str]], augmented_dataframe: DataFrame | None = None,
        seed: int | None  = None, scaler: float = 1000,
        data_augment: int = 0
    ):
        if seed is None:
            seed = numpyrand.SeedSequence().entropy # type: ignore
        self.seed = seed # type: ignore
        self.scaler = scaler
        self.data_augment = data_augment
        self.augmented_dataframe = augmented_dataframe
        labels['targets'] = [label for label in [*labels['regression targets'], *labels['class targets']]]
        labels['features'] = [label for label in [
            *labels['regular features'],
            *labels['normal features'],
            *labels['sparsed features'],
            *labels['offset features'],
            *labels['categorical features'],
            *labels['ordinal features']
        ]]
        labels['all'] = [label for label in [*labels['features'], *labels['targets']]]
        self.train_dataframe = train_dataframe
        self.test_dataframe = test_dataframe
        self.validation_dataframe = validation_dataframe
        self.original_dataframe = pandas.concat([train_dataframe, test_dataframe, validation_dataframe], axis='index')
        self.crossvalidation_dataframe = pandas.concat([train_dataframe, test_dataframe], axis='index')
        self.labels = labels
        self.__generate_encoders()
        

    def __generate_encoders(self):
        self.encoder = {label: None for label in self.labels['all']}
        for label in self.labels['sparsed features']:
            self.encoder[label] = MaxAbsScaler()
            self.encoder[label].fit(
                self.crossvalidation_dataframe.loc[:, label].to_numpy().reshape(
                    len(self.crossvalidation_dataframe), -1
                )
            )
        for label in self.labels['offset features']:
            self.encoder[label] = MinMaxScaler()
            self.encoder[label].fit(
                self.crossvalidation_dataframe.loc[:, label].to_numpy().reshape(
                    len(self.crossvalidation_dataframe), -1
                )
            )
        for label in self.labels['normal features']:
            self.encoder[label] = StandardScaler()
            self.encoder[label].fit(
                self.crossvalidation_dataframe.loc[:, label].to_numpy().reshape(
                    len(self.crossvalidation_dataframe), -1
                )
            )
        for label in self.labels['categorical features']:
            self.encoder[label] = LabelBinarizer()
            self.encoder[label].fit(
                self.crossvalidation_dataframe.loc[:, label].to_numpy().reshape(
                    len(self.crossvalidation_dataframe), -1
                )
            )
        for label in self.labels['ordinal features']:
            self.encoder[label] = OrdinalEncoder()
            self.encoder[label].fit(
                self.original_dataframe.loc[:, label].to_numpy().reshape(
                    len(self.original_dataframe), -1
                )
            )
        for label in self.labels['regular features']:
            self.encoder[label] = None
        for label in self.labels['class targets']:
            self.encoder[label] = LabelBinarizer()
            self.encoder[label].fit(
                self.original_dataframe.loc[:, label].to_numpy().reshape(
                    len(self.original_dataframe), -1
                )
            )
        for label in self.labels['regression targets']:
            self.encoder[label] = None

    def classes(self, label_type: str = 'class targets'):
        return {label: self.encoder[label].classes_ for label in self.labels[label_type]}

    def generate_tensor_dict(
        self, dataframe: DataFrame,
        augment_tensor: bool = False
    ):
        tensor_dict = {}
        for label in self.labels['all']:
            if self.encoder[label] is not None:
                tensor_dict[label] = self.encoder[label].transform(
                    dataframe.loc[:, label].to_numpy().reshape(
                        len(dataframe), -1
                    )
                )
            else:
                tensor_dict[label] = dataframe.loc[:, label].to_numpy().reshape(
                    len(dataframe), -1
                )
        tensor_dict['pandas index'] = dataframe.index.to_numpy().reshape(
            len(dataframe), -1
        )
        if augment_tensor:
            tensor_dict = noise_tensor_augment(
                tensor_dict=tensor_dict, labels=self.labels, encoder=self.encoder,
                scaler=self.scaler, data_augment=self.data_augment, seed=self.seed
            )
        return tensor_dict

    def generate_tensors(self, dataframe: DataFrame, augment_tensor: bool = False): 
        tensor_dict = self.generate_tensor_dict(dataframe=dataframe, augment_tensor=augment_tensor)
        return (
            numpy.concat([tensor_dict[label] for label in self.labels['features']], axis=1),
            numpy.concat([tensor_dict[label] for label in self.labels['targets']],  axis=1)
        )
        

    def get_tensor_sizes(self, label_type: str = 'all') -> list[int]:
        tensor_sizes = []
        for label in self.labels[label_type]:
            tensor_size = 1
            if (
                isinstance(self.encoder[label], LabelBinarizer) and
                len(self.encoder[label].classes_) > 2
            ):
                tensor_size = len(self.encoder[label].classes_)
            tensor_sizes += [tensor_size]
        return tensor_sizes

    def generate_dataframe(self, tensor: ndarray, label_type: str = 'all'): 
        tensor_sizes = self.get_tensor_sizes(label_type=label_type)
        dataframe_dict = {}
        for label, tensor in zip(
            self.labels[label_type],
            numpy.split(
                ary=numpy.atleast_2d(tensor),
                indices_or_sections=list(accumulate(tensor_sizes[:-1])),
                axis=1
            )
        ):
            if self.encoder[label] is not None:
                dataframe_dict[label] = self.encoder[label].inverse_transform(tensor).ravel()
            else:
                dataframe_dict[label] = tensor.ravel()
        return DataFrame(dataframe_dict)

    def split(self, crossvalidator: BaseCrossValidator):
        if len(self.labels['class targets']) == 1:
            generator = crossvalidator.split(
                self.crossvalidation_dataframe.loc[:, self.labels['features']],
                self.crossvalidation_dataframe.loc[:, self.labels['class targets']]
            )
        else:
            generator = crossvalidator.split(
                self.crossvalidation_dataframe.loc[:, self.labels['features']]
            )
        for train_indices, test_indices in generator:
            train_features, train_targets = self.generate_tensors(
                dataframe=self.crossvalidation_dataframe.iloc[train_indices, 0:], # type: ignore
                augment_tensor=True
            )
            test_features, test_targets = self.generate_tensors(
                dataframe=self.crossvalidation_dataframe.iloc[test_indices, 0:], # type: ignore
                augment_tensor=False
            )
            yield train_features, train_targets, test_features, test_targets

    def to_tensor_dataframes(self, dataframe: DataFrame, augment_tensor: bool = False):
        tensor_dict = self.generate_tensor_dict(
            dataframe=dataframe,
            augment_tensor=augment_tensor
        )
        features_dict = {}
        for label in self.labels['features']:
            for j in range(tensor_dict[label].shape[1]):
                features_dict[f'{label}_{j}'] = tensor_dict[label][:,j].ravel()
        regression_dict = {}
        for label in self.labels['regression targets']:
            for j in range(tensor_dict[label].shape[1]):
                regression_dict[f'{label}_{j}'] = tensor_dict[label][:,j].ravel()
        class_dataframes = {}
        for label in self.labels['class targets']:
            class_dict = {}
            for j in range(tensor_dict[label].shape[1]):
                class_dict[f'{label}_{j}'] = tensor_dict[label][:,j].ravel()
            class_dataframes[label] = DataFrame(class_dict, index=tensor_dict['pandas index'])
        return (
            DataFrame(features_dict, index=tensor_dict['pandas index']),
            DataFrame(regression_dict, index=tensor_dict['pandas index']),
            class_dataframes
        )

    def inference_function(self, pred: Tensor) -> Tensor:
        """
        Function de perdida general para los tensores, considerando
        casos de clases y regresion
        """
        torchdefault.set_defaults()
        target_sizes = self.get_tensor_sizes(label_type='targets')
        return torch.concat(
            [
                torch.atleast_2d(pred)
                if label in self.labels['regression targets']
                else torch.atleast_2d(pred.sigmoid())
                if pred.size(dim=1) == 1
                else torch.atleast_2d(pred.softmax(dim=1))
                for label, pred  in zip(
                    self.labels['targets'],
                    torch.split_with_sizes(
                        input=torch.atleast_2d(pred),
                        split_sizes=target_sizes,
                        dim=1
                    )
                )
            ],
            dim=1
        )

    def loss_fn(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Function de perdida general para los tensores, considerando
        casos de clases y regresion
        """
        torchdefault.set_defaults()
        target_sizes = self.get_tensor_sizes(label_type='targets')
        loss = torch.tensor(0.0, dtype=torch.get_default_dtype(), device=torch.get_default_device())
        for label, pred_splitted, target_splitted in zip(
            self.labels['targets'],
            torch.split_with_sizes(
                input=torch.atleast_2d(pred),
                split_sizes=target_sizes,
                dim=1
            ),
            torch.split_with_sizes(
                input=torch.atleast_2d(target),
                split_sizes=target_sizes,
                dim=1
            )
        ):
            if label in self.labels['regression targets']:
                loss += mse_loss(pred_splitted, target_splitted, reduction='mean')
            elif label in self.labels['class targets']:
                class_weights = numpy.sum(
                    target_splitted.cpu().detach().numpy().reshape(target_splitted.shape),
                    axis=0
                )
                if torch.atleast_2d(pred).size(dim=1) > 1:
                    loss += cross_entropy(
                        pred_splitted.softmax(dim=1),
                        target_splitted.softmax(dim=1),
                        reduction='mean',
                        weight=class_weights
                    )
                else:
                    loss += binary_cross_entropy_with_logits(
                        pred_splitted,
                        target_splitted,
                        reduction='mean',
                        pos_weight=torch.tensor(class_weights[0], dtype=torch.get_default_dtype(), device=torch.get_default_device())
                    )
        return loss / len(self.labels['targets'])
