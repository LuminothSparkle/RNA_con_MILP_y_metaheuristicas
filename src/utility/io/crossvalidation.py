"""
A
"""
from pathlib import Path
import pandas
from pandas import DataFrame
import matplotlib.pyplot as pyplot
import numpy
import torch
from numpy import ndarray
from sklearn.metrics import (
    ConfusionMatrixDisplay, PredictionErrorDisplay,
    accuracy_score, mean_absolute_percentage_error, mean_absolute_error,
    recall_score, roc_auc_score, precision_score, average_precision_score,
    balanced_accuracy_score, mean_squared_error, r2_score,
    d2_absolute_error_score, matthews_corrcoef, f1_score,
    log_loss, explained_variance_score
)
from src.utility.nn.lineal import set_defaults


def save_crossvalidation(
    file_path: Path,
    exists_ok: bool = True,
    **kwargs
):
    """
    A
    """
    set_defaults()
    assert exists_ok or not file_path.exists(), (
        f"El archivo {file_path} ya existe"
    )
    torch.save(kwargs, file_path)


def load_crossvalidation(
    file_path: Path,
    not_exists_ok: bool = True
):
    """
    A
    """
    set_defaults()
    assert not_exists_ok or file_path.exists(), (
        f"El archivo {file_path} no existe"
    )
    return torch.load(
        f=file_path,
        map_location=torch.get_default_device(),
        weights_only=False
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
    if percentage:
        fig, ax = pyplot.subplots(layout='constrained')
        ax.boxplot(
            numpy.concat(
                [
                    numpy.atleast_2d(data).T
                    for data in values.values()
                ],
                axis=1
            ),
            tick_labels=[
                label.capitalize()
                for label in values
            ]
        )
        ax.set(
            ylabel='Porcentaje',
            title=f'Metricas de {target_label.capitalize()}'
        )
        fig.savefig(
            file_path, transparent=True,
            pad_inches=0.1,
        )
        return fig
    else:
        fig, *axes = pyplot.subplots(1, len(values), layout='constrained')
        for ax, (label, arr) in zip(axes, values.items()):
            ax.boxplot(
                numpy.atleast_2d(arr).T,
                tick_labels=[label.capitalize()]  # type: ignore
            )
            ax.set(
                ylabel='Valor',
                title=f'Metricas de {target_label.capitalize()}'
            )
        fig.savefig(
            file_path, transparent=True,
            pad_inches=0.1,
        )
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
    cm_disp.figure_.savefig(
        file_path, transparent=True,
        bbox_inches='tight', pad_inches=0.1
    )
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
    pe_disp.figure_.savefig(
        file_path, transparent=True,
        bbox_inches='tight', pad_inches=0.1
    )
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
            y_true=targets,
            y_pred=scores
        )]
        model_metrics['average precision'] = [average_precision_score(
            y_true=tensor_targets.round().astype(int),
            y_score=scores,
            average='weighted'
        )]
        model_metrics['roc auc'] = [roc_auc_score(
            y_true=tensor_targets.round().astype(int),
            y_score=scores,
            average='weighted'
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
            f'Total time lapsed {d} days {h} hours {m} minutes {s} seconds\n'
            f'        {ms} miliseconds {mus} microseconds {ns} nanoseconds\n'
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
