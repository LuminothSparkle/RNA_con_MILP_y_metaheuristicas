"""
A
"""
import json
from collections.abc import Iterable
from pathlib import Path
import pickle
import pandas
from pandas import DataFrame
import matplotlib.pyplot as pyplot
import numpy
from numpy import ndarray
from sklearn.metrics import (
    ConfusionMatrixDisplay, PredictionErrorDisplay,
    accuracy_score, mean_absolute_percentage_error, mean_absolute_error,
    recall_score, roc_auc_score, precision_score, average_precision_score,
    balanced_accuracy_score, mean_squared_error, r2_score,
    d2_absolute_error_score, matthews_corrcoef, f1_score,
    log_loss, explained_variance_score
)
from src.utility.nn.cvdataset import CrossvalidationDataset


def save_dataset(
    dataset: CrossvalidationDataset,
    file_path: Path, subset: str | Iterable[int] | None = None,
    label_type: str | Iterable[int] | None = None,
    raw: bool = False, exists_ok: bool = True
):
    """
    A
    """
    assert exists_ok or not file_path.exists(), (
        f"El archivo {file_path} ya existe"
    )
    dataframe = dataset.to_dataframe(subset, label_type, raw)
    assert dataframe is not None, (
        "No se pudo crear el archivo"
    )
    dataframe.to_csv(
        file_path,
        index_label='ID',
        index=True,
        header=True,
        encoding='utf-8'
    )


def save_boxplot_figure(
    file_path: Path, target_label: str,
    values: dict[str, ndarray], percentage: bool = False,
    exists_ok: bool = True
):
    """
    A
    """
    assert exists_ok or not file_path.exists(), (
        f"El archivo {file_path} ya existe"
    )
    fig, ax = pyplot.subplots()
    ax.boxplot(
        numpy.concat(
            [numpy.atleast_2d(data).T for data in values.values()], axis=1),
        tick_labels=[label.capitalize() for label in values]  # type: ignore
    )
    ax.set(
        ylabel='Porcentaje' if percentage else 'Valor',
        title=f'Metricas de {target_label.capitalize()}'
    )
    fig.savefig(file_path.as_posix(), transparent=True)
    return fig


def save_confusion_matrix_figure(
    file_path: Path, target_label: str, classes: list[str],
    targets: ndarray, predictions: ndarray,
    fmt: str | None = None, normalize: str | None = None,
    exists_ok: bool = True
):
    """
    A
    """
    assert exists_ok or not file_path.exists(), (
        f"El archivo {file_path} ya existe"
    )
    cm_disp = ConfusionMatrixDisplay.from_predictions(
        y_true=targets, y_pred=predictions,
        labels=classes,
        normalize=normalize,  # type: ignore
        values_format=(
            fmt if fmt is not None
            else '>d' if normalize is None
            else '>2.3%'
        )
    )
    cm_disp.ax_.set(
        title=f'Matriz de confusion de {target_label.capitalize()}{
            f", normalizado sobre {normalize}" if normalize is not None else ""
        }',
        xlabel=f'{target_label.capitalize()} predicho',
        ylabel=f'{target_label.capitalize()} real'
    )
    cm_disp.figure_.savefig(file_path.as_posix(), transparent=True)
    return cm_disp


def save_prediction_error_figure(
    file_path: Path, target_label: str, targets: ndarray, predictions: ndarray,
    exists_ok: bool = True
):
    """
    A
    """
    assert exists_ok or not file_path.exists(), (
        f"El archivo {file_path} ya existe"
    )
    pe_disp = PredictionErrorDisplay.from_predictions(
        y_true=targets, y_pred=predictions, kind='actual_vs_predicted'
    )
    pe_disp.ax_.set(
        title=f'Regresion de {target_label.capitalize()}',
        xlabel=f'{target_label.capitalize()} predicho',
        ylabel=f'{target_label.capitalize()} real'
    )
    pe_disp.figure_.savefig(file_path.as_posix(), transparent=True)
    return pe_disp


def save_class_target_metrics(
    file_path: Path, target_label: str,
    data: list[tuple[ndarray, ndarray, ndarray, ndarray]],
    exists_ok: bool = True
):
    """
    A
    """
    assert exists_ok or not file_path.exists(), (
        f"El archivo {file_path} ya existe"
    )
    metrics_dataframes = []
    for tensor_targets, scores, targets, predictions in data:
        model_metrics = {}
        model_metrics['accuracy'] = [accuracy_score(
            y_true=targets,
            y_pred=predictions,
            normalize=True
        )]
        model_metrics['recall'] = [recall_score(
            y_true=targets,
            y_pred=predictions,
            average='weighted'
        )]
        model_metrics['precision'] = [precision_score(
            y_true=targets,
            y_pred=predictions,
            average='weighted'
        )]
        model_metrics['f1 score'] = [f1_score(
            y_true=targets,
            y_pred=predictions,
            average='weighted'
        )]
        model_metrics['balanced accuracy'] = [balanced_accuracy_score(
            y_true=targets,
            y_pred=predictions
        )]
        model_metrics['log loss'] = [log_loss(
            y_true=tensor_targets,
            y_pred=scores
        )]
        model_metrics['average precision'] = [average_precision_score(
            y_true=tensor_targets,
            y_score=scores
        )]
        model_metrics['roc auc'] = [roc_auc_score(
            y_true=tensor_targets,
            y_score=scores
        )]
        model_metrics['matthews corrcoef'] = [matthews_corrcoef(
            y_true=targets,
            y_pred=predictions
        )]
        metrics_dataframes += [DataFrame(model_metrics)]
    metrics = pandas.concat(
        metrics_dataframes, axis='index', ignore_index=True)
    metrics.to_csv(
        file_path,
        index_label=f'Metricas de clasificación para {target_label}',
        encoding='utf-8',
        header=True,
        index=True
    )
    return metrics


def save_regression_target_metrics(
    file_path: Path, target_label: str,
    data: list[tuple[ndarray, ndarray]]
):
    """
    A
    """
    metrics_dataframes = []
    for targets, predictions in data:
        model_metrics = {}
        model_metrics['absolute'] = mean_absolute_error(
            y_true=targets,
            y_pred=predictions
        )
        model_metrics['absolute percentage'] = mean_absolute_percentage_error(
            y_true=targets,
            y_pred=predictions
        )
        model_metrics['squared'] = mean_squared_error(
            y_true=targets,
            y_pred=predictions
        )
        model_metrics['r2 score'] = r2_score(
            y_true=targets,
            y_pred=predictions
        )
        model_metrics['d2 absolute'] = d2_absolute_error_score(
            y_true=targets,
            y_pred=predictions
        )
        model_metrics['explained variance'] = explained_variance_score(
            y_true=targets,
            y_pred=predictions
        )
        metrics_dataframes += [DataFrame(model_metrics)]
    metrics = pandas.concat(
        metrics_dataframes, axis='index', ignore_index=True)
    metrics.to_csv(
        file_path, index_label=f'Metricas de regresion para {target_label}',
        encoding='utf-8', header=True, index=True
    )
    return metrics


def save_log(
    class_labels_data: dict,
    regression_labels_data: dict,
    total_nanoseconds: int,
    log_path: Path,
    exists_ok: bool = True
):
    """
    A
    """
    assert exists_ok or not log_path.exists(), (
        f"El archivo {log_path} ya existe"
    )
    ns, mus = total_nanoseconds % 1000, total_nanoseconds // 1000
    mus, ms = mus % 1000, mus // 1000
    ms, s = ms % 1000, ms // 1000
    s, m = s % 60, s // 60
    m, h = m % 60, m // 60
    h, d = h % 24, h // 24
    with log_path.open('wt', encoding='utf-8') as fp:
        fp.write(
            f'Total time lapsed {d} days {h} hours {m} minutes {s} seconds'
            f'        {ms} miliseconds {mus} microseconds {ns} nanoseconds'
        )
        for labels_data in [class_labels_data, regression_labels_data]:
            for label, metrics in labels_data.items():
                for metric_name, value in metrics.items():
                    fp.write(
                        f'Mean metric {metric_name} for {label} : {value}\n')


def save_model_dataframe(
    model_data: dict,
    file_path: Path,
    exists_ok: bool = True
):
    """
    A
    """
    assert exists_ok or not file_path.exists(), (
        f"El archivo {file_path} ya existe"
    )
    DataFrame(model_data).to_csv(
        path_or_buf=file_path,
        index_label='Metricas para cada modelo',
        index=True,
        header=True,
        encoding='utf-8'
    )


def save_metrics_dataframe(
    metrics_data: dict,
    file_path: Path,
    exists_ok: bool = True
):
    """
    A
    """
    assert exists_ok or not file_path.exists(), (
        f"El archivo {file_path} ya existe"
    )
    DataFrame(metrics_data).to_csv(
        path_or_buf=file_path,
        index_label='Metricas para cada etiqueta',
        index=True,
        header=True,
        encoding='utf-8'
    )


def save_cv_json(file_path: Path, data: dict, exists_ok: bool = True):
    """
    A
    """
    assert exists_ok or not file_path.exists(), (
        f"El archivo {file_path} ya existe"
    )
    with file_path.open(mode='wt', encoding='utf-8') as fp:
        json.dump(data, fp)


def load_cv_json(file_path: Path, not_exists_ok: bool = True):
    """
    A
    """
    assert not_exists_ok or file_path.exists(), (
        f"El archivo {file_path} no existe"
    )
    with file_path.open(mode='rt', encoding='utf-8') as fp:
        data = json.load(fp)
    return data


def save_python_cv(file_path: Path, data: dict, exists_ok: bool = True):
    """
    A
    """
    assert exists_ok or not file_path.exists(), (
        f"El archivo {file_path} ya existe"
    )
    with file_path.open(mode='wb', encoding='utf-8') as fp:
        pickle.dump(data, fp)


def load_python_cv(file_path: Path, not_exists_ok: bool = True):
    """
    A
    """
    assert not_exists_ok or file_path.exists(), (
        f"El archivo {file_path} no existe"
    )
    with file_path.open(mode='rb', encoding='utf-8') as fp:
        data = pickle.load(fp)
    return data
