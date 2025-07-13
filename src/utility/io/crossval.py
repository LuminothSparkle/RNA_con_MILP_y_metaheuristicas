"""
A
"""
from pathlib import Path
import pandas
from pandas import DataFrame
import torch
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


def save_log(
    metrics: DataFrame,
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
        for label, *data in metrics.itertuples(index=True, name=None):
            for metric_name, value in (
                (metric_name, value)
                for metric_name, value in zip(data, metrics.columns)
                if pandas.isna(value)
            ):
                fp.write(
                    f'Mean metric {metric_name} for {label} : {value}\n'
                )


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
    metrics_data: DataFrame,
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
