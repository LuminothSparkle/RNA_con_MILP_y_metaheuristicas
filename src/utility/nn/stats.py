"""
A
"""
import numpy
from pandas import DataFrame
import pandas
import scipy
from sklearn.metrics import (
    accuracy_score, mean_absolute_percentage_error, mean_absolute_error,
    recall_score, roc_auc_score, precision_score, average_precision_score,
    balanced_accuracy_score, mean_squared_error, r2_score,
    d2_absolute_error_score, matthews_corrcoef, f1_score,
    log_loss, explained_variance_score
)
from src.utility.nn.lineal import LinealNN
from src.utility.nn.cvdataset import CrossvalidationDataset


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


def accuracy_comparation(
    model_a: LinealNN, model_b: LinealNN,
    dataset: CrossvalidationDataset,
    test_indices: list[int] | None = None
):
    """
    A
    """
    _, _, targets, predictions_a = dataset.prediction(
        model_a, test_indices
    )
    _, _, _, predictions_b = dataset.prediction(
        model_b, test_indices
    )
    a_over_b = numpy.sum(
        (predictions_a == targets) * (predictions_b != targets)
    )
    b_over_a = numpy.sum(
        (predictions_b == targets) * (predictions_a != targets)
    )
    a_b_good = numpy.sum(
        (predictions_a == targets) * (predictions_b == targets)
    )
    a_b_bad = numpy.sum(
        (predictions_a != targets) * (predictions_b != targets)
    )
    return a_b_good, a_over_b, b_over_a, a_b_bad


def binomial_test(
    a_over_b: int, b_over_a: int,
    expected_mean: float
):
    """
    A
    """
    total_difs = a_over_b + b_over_a
    return numpy.sum([
        scipy.special.comb(total_difs, t, exact=True)
        * (expected_mean) ** (t) * (1 - expected_mean) ** (total_difs - t)
        for t in range(a_over_b, total_difs + 1)
    ])


def mcnemar_test(a_over_b: int, b_over_a: int):
    """
    A
    """
    return (
        (abs(a_over_b - b_over_a) - 1) ** 2 / (a_over_b + b_over_a)
    )


def stats_dataframe(
    model: LinealNN, dataset: CrossvalidationDataset,
    test_indices: list[int] | None = None
):
    """
    A
    """
    class_labels = dataset.labels['class targets']
    regression_labels = dataset.labels['regression targets']
    tensor_targets, scores, targets, predictions = dataset.prediction(
        model, test_indices
    )
    model_metrics = DataFrame(
        index=[*class_labels, *regression_labels],
        columns=[*class_metrics_labels, *regression_metrics_labels]
    )
    for label in class_labels:
        model_metrics[label]['accuracy'] = (
            accuracy_score(
                y_true=targets[label],
                y_pred=predictions[label],
                normalize=True
            )
        )
        model_metrics[label]['recall'] = (
            recall_score(
                y_true=targets[label],
                y_pred=predictions[label],
                average='weighted'
            )
        )
        model_metrics[label]['precision'] = (
            precision_score(
                y_true=targets[label],
                y_pred=predictions[label],
                average='weighted'
            )
        )
        model_metrics[label]['f1 score'] = (
            f1_score(
                y_true=targets[label],
                y_pred=predictions[label],
                average='weighted'
            )
        )
        model_metrics[label]['balanced accuracy'] = (
            balanced_accuracy_score(
                y_true=targets[label],
                y_pred=predictions[label]
            )
        )
        model_metrics[label]['log loss'] = (
            log_loss(
                y_true=targets[label],
                y_pred=scores[label]
            )
        )
        model_metrics[label]['average precision'] = (
            average_precision_score(
                y_true=tensor_targets[label].round().astype(int),
                y_score=scores[label],
                average='weighted'
            )
        )
        model_metrics[label]['roc auc'] = (
            roc_auc_score(
                y_true=tensor_targets[label].round().astype(int),
                y_score=scores[label],
                average='weighted'
            )
        )
        model_metrics[label]['matthews corrcoef'] = (
            matthews_corrcoef(
                y_true=targets[label],
                y_pred=predictions[label]
            )
        )
    for label in regression_labels:
        model_metrics[label]['absolute'] = (
            mean_absolute_error(
                y_true=targets[label],
                y_pred=predictions[label]
            )
        )
        model_metrics[label]['absolute percentage'] = (
            mean_absolute_percentage_error(
                y_true=targets[label],
                y_pred=predictions[label]
            )
        )
        model_metrics[label]['squared'] = (
            mean_squared_error(
                y_true=targets[label],
                y_pred=predictions[label]
            )
        )
        model_metrics[label]['r2 score'] = (
            r2_score(
                y_true=targets[label],
                y_pred=predictions[label]
            )
        )
        model_metrics[label]['d2 absolute'] = (
            d2_absolute_error_score(
                y_true=targets[label],
                y_pred=predictions[label]
            )
        )
        model_metrics[label]['explained variance'] = (
            explained_variance_score(
                y_true=targets[label],
                y_pred=predictions[label]
            )
        )
    model_metrics.replace(numpy.nan, pandas.NA, inplace=True)
    return model_metrics


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
    mean_dataframe = DataFrame(
        numpy.nanmean(
            numpy.stack(
                [
                    numpy.atleast_3d(dataframe.to_numpy())
                    for dataframe in models_dataframes
                ],
                axis=2
            ),
            axis=2
        ),
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
