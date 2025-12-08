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


def save_boxplot_figures(
    dir_path: Path, extensions: list[str],
    boxplot_data: dict[str, DataFrame],
    backend: str
):
    """
    A
    """
    dir_path.mkdir(parents=True, exist_ok=True)
    for target_label, dataframe in boxplot_data.items():
        target_path = dir_path / target_label
        target_path.mkdir(parents=True, exist_ok=True)
        percentage_data = {
            metric: dataframe.loc[:, metric].to_numpy().reshape(-1,1)
            for metric in dataframe.columns
            if metric in display_percentage_metrics_labels
        }
        values_data = {
            metric: dataframe.loc[:, metric].to_numpy().reshape(-1,1)
            for metric in dataframe.columns
            if metric not in display_percentage_metrics_labels
        }
        for extension in extensions:
            if len(percentage_data) > 0:
                save_boxplot_figure(
                    target_path / f'percentages.{extension}',
                    target_label=target_label,
                    backend=backend,
                    values=percentage_data
                )
            if len(values_data) > 0:
                save_boxplots_figure(
                    file_path=target_path / f'values.{extension}',
                    target_label=target_label,
                    backend=backend,
                    values=values_data
                )


def save_boxplots_figure(
    file_path: Path, target_label: str,
    values: dict[str, ndarray],
    backend: str
):
    """
    A
    """
    pyplot.switch_backend(backend)
    fig, *axes = pyplot.subplots(1, len(values), layout='constrained')
    for ax, (label, arr) in zip(axes, values.items()):
        ax.boxplot(
            arr,
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
    backend: str
):
    """
    A
    """
    pyplot.switch_backend(backend)
    fig, ax = pyplot.subplots(layout='constrained')
    ax.boxplot(
        numpy.concat(
            [*values.values()],
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
):
    """
    A
    """
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
        title=f'Matriz de confusion de {target_label.capitalize()}'
        f'{f", normalizado sobre {normalize}" if normalize is not None else ""}',
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
):
    """
    A
    """
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
