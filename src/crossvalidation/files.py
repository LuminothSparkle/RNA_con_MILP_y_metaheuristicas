"""
A
"""
import argparse
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from matplotlib import pyplot
import numpy
from torch.utils.data import Subset
from src.utility.nn.stats import (
    stats_dataframe, mean_stats_dataframes,
    generate_percetages
)
from src.utility.io.figures import (
    extract_boxplot_data,
    save_boxplot_figures,
    save_confusion_matrix_figure,
    save_prediction_error_figure
)
from src.utility.io.crossval import (
    save_log, save_model_dataframe, save_metrics_dataframe,
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


def generate_cv_files(
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
    backend = 'agg'
    extensions = ['pdf']
    model_predictions = []
    datasets = []
    stats_data = []
    for model, generator, test_indices in zip(
        models, dataset_generators, models_test_indices
    ):
        dataset = CrossvalidationTensorDataset.from_generator_dict(generator)
        datasets += [dataset]
        predictions = dataset.prediction(model, test_indices)
        model_predictions += [predictions]
        stats_data += [generate_percetages(
            stats_dataframe(model, dataset, test_indices)
        )]
        mean_stats = mean_stats_dataframes(stats_data)
    boxplot_data = extract_boxplot_data(stats_data)
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures_list = []
        futures_list += [executor.submit(
            save_model_dataframe,
            model_data={
                'time lapsed': models_train_time,
                'loss': models_loss,
                'parameters': models_parameters
            },
            file_path=dir_path / f'{safe_suffix(name, "model")}.csv',
            exists_ok=exists_ok
        )]
        futures_list += [executor.submit(
            save_log,
            metrics=mean_stats,
            total_nanoseconds=numpy.sum(models_train_time),
            log_path=dir_path / f'{name}.log',
            exists_ok=exists_ok
        )]
        dataframes_path = dir_path / 'dataframes'
        stats_path = dir_path / 'stats'
        dataset_path = dir_path / 'datasets'
        models_path = dir_path / 'models'
        dataframes_path.mkdir(parents=True, exist_ok=True)
        stats_path.mkdir(parents=True, exist_ok=True)
        dataset_path.mkdir(parents=True, exist_ok=True)
        models_path.mkdir(parents=True, exist_ok=True)
        for i, model, dataset, train_indices, test_indices, stats in zip(
            [*range(len(datasets)), 'best', 'worst'],
            [
                *datasets,
                datasets[0],
                datasets[-1]
            ],
            [
                *models,
                models[0],
                models[-1]
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
            ],
            [
                *stats_data,
                stats_data[0],
                stats_data[-1]
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
                save_model,
                module=model,
                dir_path=models_path / f'model_{i}',
                name=name,
                exists_ok=exists_ok
            )]
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
            futures_list += [executor.submit(
                save_metrics_dataframe,
                metrics_data=stats,
                file_path=stats_path / (
                    f'{safe_suffix(name, f"{i}")}.csv'
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
            save_displays(
                model_predictions=model_predictions[i],
                class_encoders=datasets[i].encoders,
                class_targets=datasets[i].labels['class targets'],
                dir_path=dir_path / name_type,
                backend=backend,
                extensions=extensions,
                name=name,
                exists_ok=exists_ok
            )
        save_boxplot_figures(
            dir_path=dir_path / 'boxplots',
            boxplot_data=boxplot_data,
            extensions=extensions,
            backend=backend,
            exists_ok=exists_ok
        )
        for future_data in futures_list:
            future_data.result()


def save_displays(
    model_predictions: dict,
    class_targets: list[str],
    class_encoders: dict,
    dir_path: Path,
    extensions: list[str],
    backend: str,
    name: str = '',
    exists_ok: bool = True
):
    """
    A
    """
    pyplot.switch_backend(backend)
    dir_path.mkdir(parents=True, exist_ok=True)
    formats = {
        None: '>d', 'all': '>2.3%',
        'true': '>2.3%', 'pred': '>2.3%'
    }
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
                    save_confusion_matrix_figure(
                        figure_path,
                        label, classes, targets, pred, fmt, norm,
                        exists_ok=exists_ok
                    )
        else:
            targets, pred = data
            for extension in extensions:
                figure_path = dir_path / (
                    f'{safe_suffix(name, label)}_pe.{extension}'
                )
                save_prediction_error_figure(
                    figure_path,
                    label, targets, pred,
                    exists_ok=exists_ok
                )


def main(args: argparse.Namespace):
    """
    A
    """
    data = load_crossvalidation(
        args.load_file,
        False
    )
    generate_cv_files(
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
