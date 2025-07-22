"""
A
"""
import argparse
from argparse import ArgumentParser
from pathlib import Path
import pandas
from pandas import DataFrame
from src.utility.io.dataset import load_pytorch_dataset
from src.utility.io.model import load_model
from src.utility.nn.stats import (
    accuracy_comparation, binomial_test, stats_dataframe, weight_comparation
)
from src.utility.nn.lineal import LinealNN
from trainers.datasets import CrossvalidationDataset


def save_log(
    metrics_a: DataFrame,
    metrics_b: DataFrame,
    probability: float,
    weight_comp: list,
    log_path: Path,
    exists_ok: bool = True
):
    """
    A
    """
    assert exists_ok or not log_path.exists(), (
        f"El archivo {log_path} ya existe"
    )
    with log_path.open('wt', encoding='utf-8') as fp:
        fp.write(
            f'Probability of having more performance : {probability:.3%}\n'
        )
        for k, (mean, std) in enumerate(weight_comp):
            fp.write(
                f'Layer {k:3} mean distance and std between weights :'
                f'{mean:10}, {std:10}\n'
            )
        for (label, *data_a), (data_b) in zip(
            [*metrics_a.itertuples(index=True, name=None)],
            [*metrics_b.itertuples(index=False, name=None)]
        ):
            for metric_name, value_a, value_b in (
                (metric_name, value_a, value_b)
                for metric_name, value_a, value_b in zip(
                    metrics_a.columns, data_a, data_b
                )
                if not pandas.isna(value_a) and not pandas.isna(value_b)
            ):
                fp.write(
                    f'Metric {metric_name} for {label} : '
                    f'{value_a} vs {value_b}\n'
                )


def main(args: argparse.Namespace):
    """
    A
    """
    dataset = load_pytorch_dataset(args.dataset, False)
    model_a = load_model(args.model_a, args.name_a, False)
    model_b = load_model(args.model_b, args.name_b, False)
    dataset.crossvalidation_mode = False
    assert (
        isinstance(model_a, LinealNN) and isinstance(model_b, LinealNN)
        and isinstance(dataset, CrossvalidationDataset)
    ), "Algunos de los archivos cargados esta mal"
    _, a_over_b, b_over_a, _ = (
        accuracy_comparation(model_a, model_b, dataset)
    )
    p = binomial_test(a_over_b, b_over_a, 0.5)
    stats_a = stats_dataframe(model_a, dataset)
    stats_b = stats_dataframe(model_a, dataset)
    w_comp = weight_comparation(model_a, model_b)
    save_log(
        stats_a, stats_b, p,
        w_comp,
        args.save_path / f'{args.name_a}_vs_{args.name_b}.log',
        not args.no_overwrite
    )


if __name__ == '__main__':
    import sys
    argparser = ArgumentParser()
    argparser.add_argument('--model_a', '-ma', type=Path)
    argparser.add_argument('--model_b', '-mb', type=Path)
    argparser.add_argument('--dataset', '-ds', type=Path)
    argparser.add_argument('--save_path', '-sp', type=Path, default=Path.cwd())
    argparser.add_argument(
        '--no_overwrite', '-eo',
        action='store_true'
    )
    sys.exit(main(argparser.parse_args(sys.argv[1:])))
