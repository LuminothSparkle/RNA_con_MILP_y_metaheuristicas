"""
A
"""
import argparse
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from matplotlib import pyplot
from pandas import DataFrame
import numpy
from torch.utils.data import Subset
from src.utility.io.crossvalidation import (
    save_log, save_boxplot_figure,
    save_class_target_metrics, save_confusion_matrix_figure,
    save_prediction_error_figure,
    save_regression_target_metrics,
    save_model_dataframe, save_metrics_dataframe,
    load_crossvalidation
)
from src.utility.io.dataset import save_dataset_csv, save_pytorch_dataset
from src.utility.io.model import save_model
from src.utility.nn.lineal import LinealNN
from src.utility.nn.cvtensords import CrossvalidationTensorDataset


def safe_suffix(name: str, suffix: str):
    """
    A
    """
    if not name.endswith('_') and name != '':
        return f'{name}_{suffix}'
    return f'{name}{suffix}'


def generate_cv_data(
    dir_path: Path,
    models: list[LinealNN],
    dataset_generators: list,
    models_test_indices: list[list[int]],
    models_train_indices: list[list[int]],
    models_loss: list,
    models_train_time: list,
    models_parameters: list,
    name: str = '',
    exists_ok: bool = True,
    **kwargs
):
    """
    A
    """
    _ = kwargs  # type: ignore
    pyplot.switch_backend('agg')
    class_data, regression_data = {}, {}
    model_predictions = []
    datasets = []
    for model, generator, test_indices in zip(
        models, dataset_generators, models_test_indices
    ):
        dataset = CrossvalidationTensorDataset.from_generator_dict(generator)
        datasets += [dataset]
        predictions = dataset.prediction(model, test_indices)
        model_predictions += [predictions]
        for label in dataset.labels['class targets']:
            if label not in class_data:
                class_data[label] = []
            class_data[label] += [predictions[label]]
        for label in dataset.labels['regression targets']:
            if label not in regression_data:
                regression_data[label] = []
            regression_data[label] += [predictions[label]]
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures_list = []
        models_path = dir_path / 'models'
        futures_list += [executor.submit(
            save_models,
            models=models,
            dir_path=models_path,
            name=name,
            exists_ok=exists_ok
        )]
        dataframes_path = dir_path / 'dataframes'
        dataset_path = dir_path / 'datasets'
        for i, dataset, train_indices, test_indices in zip(
            [*range(len(datasets)), 'best', 'worst'],
            [
                *datasets,
                datasets[0],
                datasets[-1]
            ],
            [
                *models_train_indices,
                models_train_indices[0],
                models_train_indices[-1]
            ],
            [
                *models_test_indices,
                models_test_indices[0],
                models_test_indices[-1]
            ]
        ):
            model_path = dataframes_path / f'model_{i}'
            test_dataset_path = dataset_path / 'test_pt'
            train_dataset_path = dataset_path / 'train_pt'
            train_path = model_path / 'train_csv'
            test_path = model_path / 'test_csv'
            model_path.mkdir(parents=True, exist_ok=True)
            train_path.mkdir(parents=True, exist_ok=True)
            test_path.mkdir(parents=True, exist_ok=True)
            train_dataset_path.mkdir(parents=True, exist_ok=True)
            test_dataset_path.mkdir(parents=True, exist_ok=True)
            futures_list += [executor.submit(
                save_pytorch_dataset,
                dataset=Subset(dataset, train_indices),
                file_path=train_dataset_path / (
                    f'{safe_suffix(name, f"{i}")}.pt'
                ),
                exists_ok=exists_ok
            )]
            futures_list += [executor.submit(
                save_pytorch_dataset,
                dataset=Subset(dataset, test_indices),
                file_path=test_dataset_path / (
                    f'{safe_suffix(name, f"{i}")}.pt'
                ),
                exists_ok=exists_ok
            )]
            for label_type, suffix in [
                ('all', 'raw'),
                ('class targets', 'cls_tgt'),
                ('regression targets', 'reg_tgt'),
                ('features', 'ftr')
            ]:
                futures_list += [executor.submit(
                    save_dataset_csv,
                    dataset=dataset,
                    file_path=train_path / (
                        f'{safe_suffix(name, suffix)}.csv'
                    ),
                    subset=train_indices,
                    label_type=label_type,
                    raw=True,
                    exists_ok=exists_ok
                )]
                futures_list += [executor.submit(
                    save_dataset_csv,
                    dataset=dataset,
                    file_path=test_path / (
                        f'{safe_suffix(name, suffix)}.csv'
                    ),
                    subset=test_indices,
                    label_type=label_type,
                    raw=True,
                    exists_ok=exists_ok
                )]
        for name_type, i in [('best', 0), ('worst', -1)]:
            futures_list += [executor.submit(
                save_displays,
                model_predictions=model_predictions[i],
                class_encoders=datasets[i].encoders,
                class_targets=datasets[i].labels['class targets'],
                dir_path=dir_path / name_type,
                name=name,
                exists_ok=exists_ok
            )]
        boxplot_figures_path = dir_path / 'boxplots'
        boxplot_figures_path.mkdir(parents=True, exist_ok=True)
        class_labels_data, regression_labels_data, *_ = save_gen_metrics(
            class_data=class_data,
            regression_data=regression_data,
            dir_path=boxplot_figures_path,
            name=name,
            exists_ok=exists_ok
        )
        dataframes_path = dir_path / 'dataframes'
        dataframes_path.mkdir(parents=True, exist_ok=True)
        futures_list += [executor.submit(
            save_dataframes,
            model_data={
                'time lapsed': models_train_time,
                'loss': models_loss,
                'parameters': models_parameters
            },
            class_labels_data=class_labels_data,
            regression_labels_data=regression_labels_data,
            dir_path=dataframes_path,
            name=name,
            exists_ok=exists_ok
        )]
        futures_list += [executor.submit(
            save_log,
            class_labels_data=class_labels_data,
            regression_labels_data=regression_labels_data,
            total_nanoseconds=numpy.sum(models_train_time),
            log_path=dir_path / f'{name}.log',
            exists_ok=exists_ok
        )]
        for future_data in futures_list:
            future_data.result()


def save_dataframes(
    model_data: dict,
    class_labels_data: dict,
    regression_labels_data: dict,
    dir_path: Path,
    name: str = '',
    exists_ok: bool = True
):
    """
    A
    """
    dir_path.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures_list = []
        futures_list += [executor.submit(
            save_model_dataframe,
            model_data,
            dir_path / f'{safe_suffix(name, "model")}.csv',
            exists_ok
        )]
        futures_list += [executor.submit(
            save_metrics_dataframe,
            class_labels_data,
            dir_path / f'{safe_suffix(name, "class")}.csv',
            exists_ok
        )]
        futures_list += [executor.submit(
            save_metrics_dataframe,
            regression_labels_data,
            dir_path / f'{safe_suffix(name, "regression")}.csv',
            exists_ok
        )]
        for future_data in futures_list:
            future_data.result()


def save_displays(
    model_predictions: dict,
    class_targets: list[str],
    class_encoders: dict,
    dir_path: Path,
    name: str = '',
    extensions: list[str] | None = None,
    exists_ok: bool = True
):
    """
    A
    """
    dir_path.mkdir(parents=True, exist_ok=True)
    if extensions is None:
        extensions = ['pdf']
    formats = {
        None: '>d', 'all': '>2.3%',
        'true': '>2.3%', 'pred': '>2.3%'
    }
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures_list = []
        for label, data in model_predictions.items():
            if label in class_targets:
                _, _, targets, pred = data
                classes = class_encoders[label].classes_
                base_name = f'{safe_suffix(name, label)}_cm'
                for norm, fmt in formats.items():
                    for extension in extensions:
                        figure_path = (
                            dir_path / f'{base_name}_{norm}.{extension}'
                        )
                        futures_list += [executor.submit(
                            save_confusion_matrix_figure,
                            figure_path,
                            label, classes, targets, pred, fmt, norm,
                            exists_ok=exists_ok
                        )]
            else:
                targets, pred = data
                for extension in extensions:
                    figure_path = dir_path / (
                        f'{safe_suffix(name, label)}_pe.{extension}'
                    )
                    futures_list += [executor.submit(
                        save_prediction_error_figure,
                        figure_path,
                        label, targets, pred,
                        exists_ok=exists_ok
                    )]
        for future_data in futures_list:
            future_data.result()


def save_gen_metrics(
    class_data: dict, regression_data: dict,
    dir_path: Path, name: str = '',
    percentage_metrics: list[str] | None = None,
    extensions: list[str] | None = None,
    exists_ok: bool = True
):
    """
    A
    """
    dir_path.mkdir(parents=True, exist_ok=True)
    if percentage_metrics is None:
        percentage_metrics = [
            'roc auc', 'precision', 'accuracy', 'recall', 'f1 score',
            'balanced accuracy', 'average precision', 'balanced accuracy',
            'matthews corrcoef'
        ]
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures_list = []
        labels_data = []
        for data_type, data_manager in [
            (class_data,      save_class_target_metrics),
            (regression_data, save_regression_target_metrics)
        ]:
            labels_data_type = {}
            for label, data in data_type.items():
                data_metrics = data_manager(
                    dir_path / f'{safe_suffix(name, label)}.csv',
                    label, data
                )
                futures_list += [executor.submit(
                    save_boxplots,
                    dir_path=dir_path, name=name,
                    label=label, metrics=data_metrics,
                    percentage_metrics=percentage_metrics,
                    extensions=extensions,
                    exists_ok=exists_ok
                )]
                labels_data_type[label] = data_metrics.mean(axis='index')
            labels_data += [labels_data_type]
        for future_data in futures_list:
            future_data.result()
    return tuple(labels_data)


def save_boxplots(
    dir_path: Path,
    name: str,
    label: str,
    metrics: DataFrame,
    percentage_metrics: list[str] | None = None,
    extensions: list[str] | None = None,
    exists_ok: bool = True
):
    """
    A
    """
    dir_path.mkdir(parents=True, exist_ok=True)
    if extensions is None:
        extensions = ['pdf']
    if percentage_metrics is None:
        percentage_metrics = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures_list = []
        for extension in extensions:
            val_path = dir_path / (
                f'{safe_suffix(name, label)}_values.{extension}'
            )
            futures_list += [executor.submit(
                save_boxplot_figure,
                file_path=val_path,
                target_label=label,
                values={
                    metric: metrics[metric].to_numpy()
                    for metric in metrics.columns
                    if metric not in percentage_metrics
                },
                percentage=False,
                exists_ok=exists_ok
            )]
            per_path = dir_path / (
                f'{safe_suffix(name, label)}_percentages.{extension}'
            )
            futures_list += [executor.submit(
                save_boxplot_figure,
                file_path=per_path,
                target_label=label,
                values={
                    metric: metrics[metric].to_numpy()
                    for metric in metrics.columns
                    if metric in percentage_metrics
                },
                percentage=True,
                exists_ok=exists_ok
            )]
        for future_data in futures_list:
            future_data.result()


def save_models(
    models: list[LinealNN],
    dir_path: Path, name: str = '',
    exists_ok=True
):
    """_summary_

    Args:
        models (list[LinealNN]): _description_
        dir_path (Path): _description_
        name (str, optional): _description_. Defaults to ''.
        exists_ok (bool, optional): _description_. Defaults to True.
    """
    models_list = [
        *enumerate(models),
        ('model_best', models[0]),
        ('model_worst', models[-1])
    ]
    for idx, model in models_list:
        model_path = dir_path / idx
        model_path.mkdir(parents=True, exist_ok=True)
        save_model(
            model,
            model_path,
            name,
            exists_ok
        )


def main(args: argparse.Namespace):
    """
    A
    """
    data = load_crossvalidation(
        args.load_file,
        False
    )
    generate_cv_data(
        dir_path=args.save_path,
        name=args.save_name,
        exists_ok=not args.no_overwrite,
        **data
    )


if __name__ == '__main__':
    import sys
    argparser = ArgumentParser()
    argparser.add_argument(
        '--save_path', '-sp',
        default=Path.cwd()
    )
    argparser.add_argument(
        '--load_file', '-lf',
        default=Path.cwd()
    )
    argparser.add_argument(
        '--save_name', '-sn',
        default=''
    )
    argparser.add_argument('--no_overwrite', '-no', action='store_true')
    sys.exit(main(argparser.parse_args(sys.argv[1:])))
