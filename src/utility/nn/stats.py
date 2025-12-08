"""
A
"""
from itertools import accumulate
import numpy
from pandas import DataFrame
import pandas
import scipy
from sklearn.metrics import (
    accuracy_score, mean_absolute_percentage_error, mean_absolute_error,
    recall_score, roc_auc_score, precision_score, average_precision_score,
    balanced_accuracy_score, mean_squared_error, r2_score,
    d2_absolute_error_score, matthews_corrcoef, f1_score,
    log_loss, explained_variance_score, confusion_matrix
)
from statsmodels.stats.contingency_tables import mcnemar
from utility.nn.lineal import LinealNN
from utility.nn.dataset import CsvDataset
import utility.nn.torchdefault as torchdefault



class_metrics_labels = [
    'accuracy', 'recall', 'precision', 'f1 score', 'balanced accuracy',
    'log loss', 'average precision', 'roc auc', 'matthews corrcoef'
]

percentages_labels = [
    'absolute percentage', 'accuracy', 'r2 score', 'explained variance',
    'd2 absolute', 'recall', 'precision', 'f1 score',
    'balanced accuracy', 'average precision', 'roc auc', 'matthews corrcoef'
]

regression_metrics_labels = [
    'absolute', 'absolute percentage', 'squared', 'r2 score', 'd2 absolute',
    'explained variance'
]


def weight_comparation(
    model_a: LinealNN, model_b: LinealNN
):
    """
    A
    """
    stats = []
    for wa, wb in zip(model_a.get_weights(), model_b.get_weights()):
        diff = wa - wb
        stats += [(numpy.mean(diff), numpy.std(diff))]
    return stats

def compare_archs(model_a: LinealNN, model_b: LinealNN, dataset: CsvDataset):
    stats = {}
    prediction_a = prediction_dataframes(model=model_a, dataset=dataset)
    prediction_b = prediction_dataframes(model=model_b, dataset=dataset)
    stats_a = stats_dataframes(
        prediction_dataframe=prediction_a['prediction'],
        classes_dataframe=prediction_a['classes'],
        class_labels=dataset.labels['class targets'],
        regression_labels=dataset.labels['regression targets']
    )
    stats_b = stats_dataframes(
        prediction_dataframe=prediction_b['prediction'],
        classes_dataframe=prediction_b['classes'],
        class_labels=dataset.labels['class targets'],
        regression_labels=dataset.labels['regression targets']
    )
    stats['class']      = compare_dataframes(stats_b['class'], stats_a['class'])
    stats['regression'] = compare_dataframes(stats_b['regression'], stats_a['regression'])
    for label in dataset.labels['class targets']:
        a_right = (
            prediction_a['prediction'][:, f'{label} target labels'] ==
            prediction_a['prediction'][:, f'{label} predicted labels']
        ).to_numpy().ravel()
        b_right = (
            prediction_b['prediction'][:, f'{label} target labels'] ==
            prediction_b['prediction'][:, f'{label} predicted labels']
        ).to_numpy().ravel()
        a_over_b = numpy.logical_and(a_right, numpy.logical_not(b_right))
        b_over_a = numpy.logical_and(b_right, numpy.logical_not(a_right))
        ab_wrong = numpy.logical_and(numpy.logical_not(a_right), numpy.logical_not(b_right))
        ab_right = numpy.logical_and(b_right, a_right)
        bunch = mcnemar(
            table=[[ab_right, a_over_b], [b_over_a, ab_wrong]],
            exact=True,
            correction=True
        )
        stats['class'][label, 'mcnemar p'] = bunch.p
        stats['class'][label, 'mcnemar statistic'] = bunch.statistic
    return stats

def prediction_dataframes(
    model: LinealNN, dataset: CsvDataset
):
    target_labels = dataset.labels['targets']
    predicted_scores = {label:[] for label in target_labels}
    target_scores = {label:[] for label in target_labels}
    predicted_labels = {label:[] for label in target_labels}
    target_labels = {label:[] for label in target_labels}
    for features, targets in torchdefault.sequential_dataloader(
        dataset=torchdefault.tensor_dataset(*dataset.generate_tensors(
            dataframe=dataset.validation_dataframe, augment_tensor=False
        ))
    ):
        tensor_sizes = dataset.get_tensor_sizes('targets')
        predicted_tensor = dataset.inference_function(model(features)).cpu().detach().numpy()
        target_tensor = targets.cpu().detach().numpy()
        predicted_dataframe = dataset.generate_dataframe(
            predicted_tensor,
            'targets'
        )
        target_dataframe = dataset.generate_dataframe(
            target_tensor,
            'targets'
        )
        for label, predicted, target in zip(
            target_labels,
            numpy.split(predicted_tensor, list(accumulate(tensor_sizes[:-1]))),
            numpy.split(target_tensor,    list(accumulate(tensor_sizes[:-1]))),
        ):
            predicted_scores[label] += [predicted]
            target_scores[label]    += [target]
            predicted_labels[label] += [predicted_dataframe[label].to_numpy().ravel()]
            target_labels[label]    += [target_dataframe[label].to_numpy().ravel()]
    df_prediction = dataset.validation_dataframe[dataset.labels['features']]
    df_classes = DataFrame()
    for label in target_labels:
        target_labels = numpy.concat(target_labels[label], axis=0)
        pred_labels = numpy.concat(predicted_labels[label], axis=0)
        df_prediction.insert(0, f'{label} target labels', target_labels)
        df_prediction.insert(0, f'{label} predicted labels', pred_labels)
        if label in dataset.labels['class targets'] and len(dataset.encoder[label].classes_) > 2:
            for class_label in dataset.encoder[label].classes_:
                df_prediction.insert(0, f'{label} target scores {class_label}', numpy.concat(target_scores[label], axis=0))
                df_prediction.insert(0, f'{label} predicted scores {class_label}', numpy.concat(predicted_scores[label], axis=0))    
        else:
            df_prediction.insert(0, f'{label} target scores', numpy.concat(target_scores[label], axis=0))
            df_prediction.insert(0, f'{label} predicted scores', numpy.concat(predicted_scores[label], axis=0))
        if label in dataset.labels['class targets']:
            df_classes.insert(0, f'{label} classes', dataset.encoder[label].classes_)
            df_classes.insert(0, f'{label} predicted', numpy.sum(
                dataset.encoder[label].classes_.reshape(1,-1) == pred_labels.reshape((-1,1)),
                axis=0
            ).squeeze())
            df_classes.insert(0, f'{label} target', numpy.sum(
                dataset.encoder[label].classes_.reshape(1,-1) == target_labels.reshape((-1,1)),
                axis=0
            ).squeeze())
    return {'prediction': df_prediction, 'classes': df_classes}

def stats_dataframes(
    prediction_dataframe: DataFrame, classes_dataframe: DataFrame,
    class_labels: list[str], regression_labels: list[str]
):
    """
    A
    """
    predicted_scores = {}
    target_scores = {}
    for label in class_labels:
        classes = classes_dataframe[f'{label} classes'].dropna().to_numpy()
        if len(classes) > 2:
            predicted_score = prediction_dataframe.loc[
                :, [f'{label} predicted scores {class_label}' for class_label in classes]
            ].to_numpy()
            target_score = prediction_dataframe.loc[
                :, [f'{label} target scores {class_label}' for class_label in classes]
            ].to_numpy()
        else:
            predicted_score = prediction_dataframe.loc[:, f'{label} predicted scores']
            target_score = prediction_dataframe.loc[:, f'{label} target scores']
        predicted_scores[label] = predicted_score
        target_scores[label] = target_score
    for label in regression_labels:
        predicted_scores[label] = prediction_dataframe.loc[:, f'{label} predicted scores']
        target_scores[label] = prediction_dataframe.loc[:, f'{label} target scores']
    predicted_labels = {
        label: prediction_dataframe.loc[:, f'{label} predicted labels'].to_numpy()
        for label in [*class_labels, *regression_labels]
    }
    target_labels = {
        label: prediction_dataframe.loc[:, f'{label} target labels'].to_numpy()
        for label in [*class_labels, *regression_labels]
    }
    class_metrics = DataFrame(
        index=class_labels,
        columns=class_metrics_labels
    )
    regression_metrics = DataFrame(
        index=regression_labels,
        columns=regression_metrics_labels
    )
    for label in class_labels:
        class_metrics.at[label, 'accuracy'] = (
            accuracy_score(
                y_true=target_labels[label],
                y_pred=predicted_labels[label],
                normalize=True
            )
        )
        class_metrics.at[label, 'recall'] = (
            recall_score(
                y_true=target_labels[label],
                y_pred=predicted_labels[label],
                average='weighted'
            )
        )
        class_metrics.at[label, 'precision'] = (
            precision_score(
                y_true=target_labels[label],
                y_pred=predicted_labels[label],
                average='weighted'
            )
        )
        class_metrics.at[label, 'f1 score'] = (
            f1_score(
                y_true=target_labels[label],
                y_pred=predicted_labels[label],
                average='weighted'
            )
        )
        class_metrics.at[label, 'balanced accuracy'] = (
            balanced_accuracy_score(
                y_true=target_labels[label],
                y_pred=predicted_labels[label]
            )
        )
        class_metrics.at[label, 'matthews corrcoef'] = (
            matthews_corrcoef(
                y_true=target_labels[label],
                y_pred=predicted_labels[label]
            )
        )
        class_metrics.at[label, 'log loss'] = (
            log_loss(
                y_true=target_scores[label],
                y_pred=predicted_scores[label]
            )
        )
        class_metrics.at[label, 'average precision'] = (
            average_precision_score(
                y_true=target_scores[label],
                y_score=predicted_scores[label],
                average='weighted'
            )
        )
        class_metrics.at[label, 'roc auc'] = (
            roc_auc_score(
                y_true=target_scores[label],
                y_score=predicted_scores[label],
                average='weighted'
            )
        )
    for label in regression_labels:
        regression_metrics.at[label, 'absolute'] = (
            mean_absolute_error(
                y_true=target_labels[label],
                y_pred=predicted_labels[label]
            )
        )
        regression_metrics.at[label, 'absolute percentage'] = (
            mean_absolute_percentage_error(
                y_true=target_labels[label],
                y_pred=predicted_labels[label]
            )
        )
        regression_metrics.at[label, 'squared'] = (
            mean_squared_error(
                y_true=target_labels[label],
                y_pred=predicted_labels[label]
            )
        )
        regression_metrics.at[label, 'r2 score'] = (
            r2_score(
                y_true=target_labels[label],
                y_pred=predicted_labels[label]
            )
        )
        regression_metrics.at[label, 'd2 absolute'] = (
            d2_absolute_error_score(
                y_true=target_labels[label],
                y_pred=predicted_labels[label]
            )
        )
        regression_metrics.at[label, 'explained variance'] = (
            explained_variance_score(
                y_true=target_labels[label],
                y_pred=predicted_labels[label]
            )
        )
    return {'class': class_metrics, 'regression': regression_metrics}

def label_dataframes(metrics_dataframes: list[DataFrame], labels: list[str]):
    return {
        label: pandas.concat(
            objs=[dataframe.loc[[label],:] for dataframe in metrics_dataframes],
            axis='index',
            join='outer',
            ignore_index=True
        )
        for label in labels
    }

def confusion_matrix_dataframes(
    predictions_dataframe: DataFrame,
    classes_dataframe: DataFrame,
    class_labels: list[str]
):
    data_dict = {}
    for label in class_labels:
        classes = classes_dataframe[f'{label} classes'].dropna().to_numpy()
        target = predictions_dataframe.loc[:,f'{label} target labels'].to_numpy()
        predicted = predictions_dataframe.loc[:,f'{label} predicted labels'].to_numpy()
        data_dict[label] = {}
        data_dict[label]['none'] = DataFrame(
            data=confusion_matrix(
                y_pred=predicted, y_true=target,
                normalize=None,   labels=classes
            ),
            columns=classes,
            index=classes
        )
        data_dict[label]['pred'] = DataFrame(
            data=confusion_matrix(
                y_pred=predicted, y_true=target,
                normalize='pred',   labels=classes
            ),
            columns=classes,
            index=classes
        )
        data_dict[label]['true'] = DataFrame(
            data=confusion_matrix(
                y_pred=predicted, y_true=target,
                normalize='true',   labels=classes
            ),
            columns=classes,
            index=classes
        )
        data_dict[label]['all'] = DataFrame(
            data=confusion_matrix(
                y_pred=predicted, y_true=target,
                normalize='all',   labels=classes
            ),
            columns=classes,
            index=classes
        )
    return data_dict


def generate_percetages(data: DataFrame):
    """
    A
    """
    data = data.copy()
    for metric_label in percentages_labels:
        data[metric_label, :] = data[metric_label, :] * 100
    data.replace(numpy.nan, pandas.NA, inplace=True)
    return data


def mean_stats_dataframes(models_dataframes: list[DataFrame]):
    """
    A
    """
    if len(models_dataframes) == 0:
        return DataFrame()
    data_array = numpy.concat(
        [
            numpy.atleast_3d(dataframe.to_numpy(na_value=numpy.nan))
            for dataframe in models_dataframes
        ],
        axis=2
    )
    data_nan = numpy.all(a=numpy.logical_not(numpy.isnan(data_array.astype(float))), axis=2)
    df_data = numpy.full(data_nan.shape, fill_value=numpy.nan)
    df_data[data_nan] = numpy.nanmean(
        a=data_array[data_nan],
        axis=1,
    )
    mean_dataframe = DataFrame(
        data=df_data,
        index=models_dataframes[0].index,
        columns=models_dataframes[0].columns
    )
    mean_dataframe.replace(numpy.nan, pandas.NA, inplace=True)
    return mean_dataframe


def compare_dataframes(data: DataFrame, reference: DataFrame):
    """
    A
    """
    data = DataFrame(
        data=(data.to_numpy() / reference.to_numpy() * 100),
        index=data.index,
        columns=data.columns
    )
    data.replace(numpy.nan, pandas.NA, inplace=True)
    return data
