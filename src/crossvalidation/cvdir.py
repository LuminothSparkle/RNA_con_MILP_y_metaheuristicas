"""
A
"""
import argparse
from argparse import ArgumentParser
from pathlib import Path
from pandas import DataFrame
import matplotlib.pyplot as pyplot
import numpy
from src.utility.io.cv import (
    save_cv_json, save_log, save_dataset, save_boxplot_figure,
    save_class_target_metrics, save_confusion_matrix_figure,
    save_prediction_error_figure, save_python_cv,
    save_regression_target_metrics,
    save_model_dataframe, save_metrics_dataframe,
    load_cv_json, load_python_cv
)
from src.utility.io.model import save_model
from src.utility.nn.lineal import LinealNN


def safe_suffix(name: str, suffix: str):
    """
    A
    """
    if not name.endswith('_') and name != '':
        return f'{name}_{suffix}'
    return f'{name}{suffix}'


def save_crossvalidation(
    dir_path: Path, cv_data: dict,
    name: str = '', exists_ok: bool = True
):
    """
    A
    """
    json_path = dir_path / f'{name}.json'
    save_cv_json(json_path, cv_data, exists_ok=exists_ok)
    pkl_path = dir_path / f'{safe_suffix(name, "cv")}.pkl'
    save_python_cv(pkl_path, cv_data, exists_ok=exists_ok)


def load_crossvalidation(
    dir_path: Path,
    name: str = '',
    not_exists_ok: bool = True
):
    """
    A
    """
    try:
        json_path = dir_path / f'{name}.json'
        data = load_cv_json(json_path, not_exists_ok=not_exists_ok)
        assert isinstance(data, dict)
    except AssertionError as ae:
        print(ae)
        data = None
    try:
        pkl_path = dir_path / f'{safe_suffix(name, "cv")}.pkl'
        data = load_python_cv(pkl_path, not_exists_ok=not_exists_ok)
        assert isinstance(data, dict)
    except AssertionError as ae:
        print(ae)
        data = None
    assert not_exists_ok or data is not None
    return data


def generate_cv_data(
    dir_path: Path, cv_data: dict,
    name: str = '', exists_ok: bool = True
):
    """
    A
    """
    models_path = dir_path / 'models'
    models_path.mkdir(parents=True, exist_ok=True)
    save_models(
        models=cv_data['model'],
        dir_path=models_path,
        name=name,
        exists_ok=exists_ok
    )
    dataframes_path = dir_path / 'dataframes'
    dataframes_path.mkdir(parents=True, exist_ok=True)
    for i, dataset, train_indexes, test_indexes in zip(
        [*range(len(cv_data['dataset'])), 'best', 'worst'],
        [*cv_data['dataset'], cv_data['dataset'][0], cv_data['dataset'][-1]],
        [
            *cv_data['train indexes'],
            cv_data['train indexes'][0],
            cv_data['train indexes'][-1]
        ],
        [
            *cv_data['test indexes'],
            cv_data['test indexes'][0],
            cv_data['test indexes'][-1]
        ]
    ):
        dir_path = dataframes_path / f'model_{i}'
        dir_path.mkdir(parents=True, exist_ok=True)
        for label_type, suffix in [
            ('all', 'raw'),
            ('class targets', 'cls_tgt'),
            ('regression targets', 'reg_tgt'),
            ('features', 'ftr')
        ]:
            save_dataset(
                dataset=dataset,
                file_path=dir_path / (
                    f'{safe_suffix(name, 'train')}_{suffix}.csv'
                ),
                subset=train_indexes,
                label_type=label_type,
                raw=True,
                exists_ok=exists_ok
            )
            save_dataset(
                dataset=dataset,
                file_path=dir_path / (
                    f'{safe_suffix(name, 'test')}_{suffix}.csv'
                ),
                subset=test_indexes,
                label_type=label_type,
                raw=True,
                exists_ok=exists_ok
            )
    class_data, regression_data = {}, {}
    model_predictions = []
    for model, dataset, test_dataloader in zip(
        cv_data['model'], cv_data['dataset'], cv_data['test dataloader']
    ):
        predictions = dataset.prediction(model, test_dataloader)
        model_predictions += [predictions]
        for label in dataset.labels['class targets']:
            if label not in class_data:
                class_data[label] = []
            class_data[label] += [predictions[label]]
        for label in dataset.labels['regression targets']:
            if label not in regression_data:
                regression_data[label] = []
            regression_data[label] += [predictions[label]]
    boxplot_figures_path = dir_path / 'boxplots'
    boxplot_figures_path.mkdir(parents=True, exist_ok=True)
    class_labels_data, regression_labels_data, *_ = save_gen_metrics(
        class_data=class_data,
        regression_data=regression_data,
        dir_path=boxplot_figures_path,
        name=name,
        exists_ok=exists_ok
    )
    for name_type, i in {'best': 0, 'worst': -1}.items():
        save_displays(
            model_predictions=model_predictions[i],
            class_encoders=cv_data['dataset'][i].encoders,
            class_targets=cv_data['dataset'][i].labels['class targets'],
            dir_path=dir_path / name_type,
            name=name,
            exists_ok=exists_ok
        )
    dataframes_path = dir_path / 'dataframes'
    dataframes_path.mkdir(parents=True, exist_ok=True)
    save_dataframes(
        model_data={
            'time lapsed': cv_data['train time'],
            'loss': cv_data['loss'],
            'parameters': cv_data['model'].get_total_parameters()
        },
        class_labels_data=class_labels_data,
        regression_labels_data=regression_labels_data,
        dir_path=dataframes_path,
        name=name,
        exists_ok=exists_ok
    )
    save_log(
        class_labels_data=class_labels_data,
        regression_labels_data=regression_labels_data,
        total_nanoseconds=numpy.sum(cv_data['train time']),
        log_path=dir_path / f'{name}.log',
        exists_ok=exists_ok
    )


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
    model_path = dir_path / f'{safe_suffix(name, "model")}.csv'
    save_model_dataframe(model_data, model_path, exists_ok)
    for labels_data, labels_metrics_path in [
        (class_labels_data, dir_path / f'{safe_suffix(name, "class")}.csv',),
        (regression_labels_data, dir_path / (
            f'{safe_suffix(name, "regression")}.csv'
        ),)
    ]:
        save_metrics_dataframe(labels_data, labels_metrics_path, exists_ok)


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
        extensions = ['eps', 'png', 'pdf', 'svg', 'tiff', 'ps', 'raw']
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
                    figure_path = dir_path / f'{base_name}_{norm}.{extension}'
                    save_confusion_matrix_figure(
                        figure_path,
                        label, classes, targets, pred, fmt, norm,
                        exists_ok=exists_ok
                    )
                    pyplot.close()
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
                pyplot.close()


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
            save_boxplots(
                dir_path=dir_path, name=name,
                label=label, metrics=data_metrics,
                percentage_metrics=percentage_metrics,
                extensions=extensions,
                exists_ok=exists_ok
            )
            labels_data_type[label] = data_metrics.mean(axis='index')
        labels_data += [labels_data_type]
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
        extensions = ['eps', 'png', 'pdf', 'svg', 'tiff', 'ps', 'raw']
    if percentage_metrics is None:
        percentage_metrics = []
    for metric in metrics.columns:
        if metric not in percentage_metrics:
            for extension in extensions:
                file_path = dir_path / (
                    f'{safe_suffix(name, label)}_{metric}.{extension}'
                )
                save_boxplot_figure(
                    file_path=file_path,
                    target_label=label,
                    values={metric: metrics[metric].to_numpy()},
                    percentage=False,
                    exists_ok=exists_ok
                )
                pyplot.close()
        for extension in extensions:
            file_path = dir_path / (
                f'{safe_suffix(name, label)}_percentages.{extension}'
            )
            save_boxplot_figure(
                file_path=file_path,
                target_label=label,
                values={
                    metric: metrics[metric].to_numpy()
                    for metric in percentage_metrics
                },
                percentage=True,
                exists_ok=exists_ok
            )
            pyplot.close()


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
    dir_path.mkdir(parents=True, exist_ok=True)
    name_dict = {
        safe_suffix(name, f'{i}'): models[i]
        for i in range(len(models))
    }
    name_dict[safe_suffix(name, 'best')] = models[0]
    name_dict[safe_suffix(name, 'worst')] = models[-1]
    for name, model in name_dict.items():
        save_model(
            module=model,
            dir_path=dir_path,
            name=name,
            exists_ok=exists_ok
        )


def main(args: argparse.Namespace):
    """
    A
    """
    data = load_crossvalidation(args.load_path, args.name, False)
    if isinstance(data, dict):
        generate_cv_data(
            args.save_path,
            data,
            args.name,
            not args.no_overwrite
        )


if __name__ == '__main__':
    import sys
    argparser = ArgumentParser()
    argparser.add_argument(
        '--save_path', '-sp',
        default=Path.cwd()
    )
    argparser.add_argument(
        '--load_path', '-lp',
        default=Path.cwd()
    )
    argparser.add_argument(
        '--name', '-lp',
        default=''
    )
    argparser.add_argument('--no_overwrite', '-no', action='store_true')
    sys.exit(main(argparser.parse_args(sys.argv[1:])))
