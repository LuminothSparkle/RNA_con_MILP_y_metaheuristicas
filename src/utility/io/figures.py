"""
A
"""
from pathlib import Path
import matplotlib.pyplot as pyplot
from pandas import DataFrame
import numpy
from numpy import ndarray
from sklearn.metrics import ConfusionMatrixDisplay, PredictionErrorDisplay


display_percentage_metrics_labels = [
    'accuracy', 'recall', 'precision', 'f1 score',
    'balanced accuracy', 'average precision', 'roc auc', 'matthews corrcoef'
]


def extract_boxplot_data(dataframes: list[DataFrame]):
    """
    A
    """
    data = numpy.stack(
        [
            numpy.atleast_3d(dataframe.to_numpy())
            for dataframe in dataframes
        ],
        axis=2
    )
    return {
        label: {
            metric: data[i, j, :].squeeze()
            for j, metric in enumerate(dataframes[0].columns)
        }
        for i, label in enumerate(dataframes[0].index)
    }


def save_boxplot_figures(
    dir_path: Path, extensions: list[str],
    boxplot_data: dict[str, dict[str, ndarray]],
    backend: str,
    exists_ok: bool = True
):
    """
    A
    """
    for target_label, data in boxplot_data.items():
        target_path = dir_path / target_label
        percentage_data = {
            metric: arr
            for metric, arr in data.items()
            if metric in display_percentage_metrics_labels
        }
        values_data = {
            metric: arr
            for metric, arr in data.items()
            if metric not in display_percentage_metrics_labels
        }
        for extension in extensions:
            save_boxplot_figure(
                target_path / f'percentages.{extension}',
                target_label=target_label,
                backend=backend,
                values=percentage_data,
                exists_ok=exists_ok
            )
            save_boxplots_figure(
                file_path=target_path / f'values.{extension}',
                target_label=target_label,
                backend=backend,
                values=values_data,
                exists_ok=exists_ok
            )


def save_boxplots_figure(
    file_path: Path, target_label: str,
    values: dict[str, ndarray],
    backend: str,
    exists_ok: bool = True
):
    """
    A
    """
    assert exists_ok or not file_path.exists(), (
        f"El archivo {file_path} ya existe"
    )
    pyplot.switch_backend(backend)
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


def save_boxplot_figure(
    file_path: Path, target_label: str,
    values: dict[str, ndarray],
    backend: str,
    exists_ok: bool = True
):
    """
    A
    """
    assert exists_ok or not file_path.exists(), (
        f"El archivo {file_path} ya existe"
    )
    pyplot.switch_backend(backend)
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
        ylabel='Porcentajes',
        title=f'Metricas de {target_label.capitalize()}'
    )
    fig.savefig(
        file_path, transparent=True,
        pad_inches=0.1,
    )


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


def save_prediction_error_figure(
    file_path: Path, target_label: str,
    targets: ndarray, predictions: ndarray,
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
