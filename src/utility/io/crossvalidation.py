"""
A
"""
from pathlib import Path
import numpy
import pandas
from pandas import DataFrame
from utility.nn.stats import (
    stats_dataframes, prediction_dataframes,
    label_dataframes, mean_stats_dataframes,
    confusion_matrix_dataframes
)
from utility.nn.crossvalidation import CrossvalidationNN
from utility.nn.trainer import TrainerNN
from utility.nn.dataset import CsvDataset
from utility.io.figures import (
    save_boxplot_figures, save_confusion_matrix_figure,
    save_prediction_error_figure
)
import utility.nn.torchdefault as torchdefault


def save_log(
    metrics: DataFrame,
    total_nanoseconds: int,
    log_path: Path
):
    """
    A
    """
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

def save_crossvalidation_data(crossvalidation: CrossvalidationNN, dir_path: Path):
    losses = [
        trainer.states['fit']['loss']
        for trainer in crossvalidation.trainers
        if trainer.states['fit']['loss'] is not None
    ]
    trainers = [
        trainer
        for trainer in crossvalidation.trainers
        if trainer.states['fit']['loss'] is not None
    ]
    sorted_losses = numpy.argsort(losses)
    total_time = numpy.sum([trainer.train_time for trainer in crossvalidation.trainers])
    prediction_dfs = []
    classes_dfs = []
    class_stats = []
    regression_stats = []
    cm_dfs = []
    for trainer in [trainers[i] for i in sorted_losses]:
        dfs = prediction_dataframes(model=trainer.model, dataset=crossvalidation.dataset)
        prediction_dfs += [dfs['prediction']]
        classes_dfs += [dfs['classes']]
        stats_dict = stats_dataframes(
            prediction_dataframe=dfs['prediction'],
            classes_dataframe=dfs['classes'],
            class_labels=crossvalidation.dataset.labels['class targets'],
            regression_labels=crossvalidation.dataset.labels['regression targets']
        )
        cm_dfs += [confusion_matrix_dataframes(
            dfs['prediction'], dfs['classes'],
            crossvalidation.dataset.labels['class targets']
        )]
        class_stats += [stats_dict['class']]
        regression_stats += [stats_dict['regression']]
    class_dfs = label_dataframes(class_stats, crossvalidation.dataset.labels['class targets'])
    regression_dfs = label_dataframes(regression_stats, crossvalidation.dataset.labels['regression targets'])
    class_mean_stats = mean_stats_dataframes(class_stats)
    regression_mean_stats = mean_stats_dataframes(regression_stats)
    dataframes = []
    set_indexes = []
    tensor_data = {}
    tensor_data['label'] = crossvalidation.dataset.labels['all']
    tensor_data['type']  = [
        label for label in ['features', 'regression targets', 'class targets']
        for _ in range(len(crossvalidation.dataset.labels[label]))
    ]
    tensor_data['sizes'] = crossvalidation.dataset.get_tensor_sizes('all')
    tensor_dataframe = DataFrame(tensor_data)
    trainers = []
    for i in sorted_losses:
        validation_data = numpy.concat(
            crossvalidation.dataset.generate_tensors(dataframe=crossvalidation.dataset.validation_dataframe),
            axis=1
        )
        test_data = numpy.concat([
            crossvalidation.tensors[i][label].cpu().detach().numpy()
            for label in ['test_features','test_targets']
        ], axis=1)
        train_data = numpy.concat([
            crossvalidation.tensors[i][label].cpu().detach().numpy()
            for label in ['train_features','train_targets']
        ], axis=1)
        indices = numpy.cumsum([train_data.shape[0], test_data.shape[0], validation_data.shape[0]])
        indices_df = DataFrame(
            columns=['train', 'test', 'crossvalidation', 'validation'],
            index=[*range(indices[2])]
        )
        indices_df.loc[[*range(train_data.shape[0])], 'train'] = [*range(0, indices[0])]
        indices_df.loc[[*range(test_data.shape[0])], 'test'] = [*range(indices[0], indices[1])]
        indices_df.loc[[*range(indices[1])], 'crossvalidation'] = [*range(0, indices[1])]
        indices_df.loc[[*range(validation_data.shape[0])], 'validation'] = [*range(indices[1], indices[2])]
        set_indexes += [indices_df]
        dataframes += [DataFrame(numpy.concat((train_data, test_data, validation_data), axis=0))]
        trainers += [crossvalidation.trainers[i]]
    torchdefault.save(state_dict=crossvalidation.dataset.state_dict(), f=dir_path / 'dataset_state_dict.pt')
    data_dir_path = dir_path / 'data'
    data_dir_path.mkdir(parents=True,exist_ok=True)
    save_log(metrics=class_mean_stats, total_nanoseconds=total_time, log_path=data_dir_path / 'class.log')
    save_log(metrics=regression_mean_stats, total_nanoseconds=total_time, log_path=data_dir_path / 'regression.log')
    class_mean_stats.to_csv(path_or_buf=data_dir_path / 'class_stats.csv')
    regression_mean_stats.to_csv(path_or_buf=data_dir_path / 'regression_stats.csv')
    save_boxplot_figures(
        boxplot_data=class_dfs,
        dir_path=data_dir_path / 'class_boxplots',
        extensions=['pdf'],
        backend='pdf'
    )
    save_boxplot_figures(
        boxplot_data=regression_dfs,
        dir_path=data_dir_path / 'regression_boxplots', 
        extensions=['pdf'], 
        backend='pdf'
    )
    class_dir_path = dir_path / 'class'
    class_dir_path.mkdir(parents=True, exist_ok=True)
    for label in crossvalidation.dataset.labels['class targets']:
        class_dfs[label].to_csv(path_or_buf=class_dir_path / f'{label}_stats.csv', index=False)
    regression_dir_path = dir_path / 'regression'
    regression_dir_path.mkdir(parents=True, exist_ok=True)
    for label in crossvalidation.dataset.labels['regression targets']:
        regression_dfs[label].to_csv(path_or_buf=regression_dir_path / f'{label}_stats.csv', index=False)
    for i in [-1, 0]:
        model_dir_path = dir_path / f'model {(i + len(sorted_losses)) % len(sorted_losses)}'
        save_tensor_dataset(
            tensor_df=tensor_dataframe,
            dataset_df=dataframes[i],
            indexes_df=set_indexes[i],
            dir_path=model_dir_path
        )
        save_trainer(trainers[i], dir_path=model_dir_path)
        for label in crossvalidation.dataset.labels['class targets']:
            label_path = model_dir_path / label
            label_path.mkdir(parents=True, exist_ok=True)
            cm_dfs[i][label]['none'].to_csv(path_or_buf=label_path / 'none.csv')
            cm_dfs[i][label]['all'].to_csv(path_or_buf=label_path / 'all.csv')
            cm_dfs[i][label]['true'].to_csv(path_or_buf=label_path / 'true.csv')
            cm_dfs[i][label]['pred'].to_csv(path_or_buf=label_path / 'pred.csv')
            save_confusion_matrix_figure(
                file_path=label_path / 'none.pdf',
                classes=classes_dfs[i][f'{label} classes'].dropna().to_numpy(),
                target_label=label,
                predictions=prediction_dfs[i][f'{label} predicted labels'].to_numpy(),
                targets=prediction_dfs[i][f'{label} target labels'].to_numpy(),
                normalize=None,
                fmt=''
            )
            save_confusion_matrix_figure(
                file_path=label_path / 'all.pdf',
                classes=classes_dfs[i][f'{label} classes'].dropna().to_numpy(),
                target_label=label,
                predictions=prediction_dfs[i][f'{label} predicted labels'].to_numpy(),
                targets=prediction_dfs[i][f'{label} target labels'].to_numpy(),
                normalize='all',
                fmt=''
            )
            save_confusion_matrix_figure(
                file_path=label_path / 'true.pdf',
                classes=classes_dfs[i][f'{label} classes'].dropna().to_numpy(),
                target_label=label,
                predictions=prediction_dfs[i][f'{label} predicted labels'].to_numpy(),
                targets=prediction_dfs[i][f'{label} target labels'].to_numpy(),
                normalize='true',
                fmt=''
            )
            save_confusion_matrix_figure(
                file_path=label_path / 'pred.pdf',
                classes=classes_dfs[i][f'{label} classes'].dropna().to_numpy(),
                target_label=label,
                predictions=prediction_dfs[i][f'{label} predicted labels'].to_numpy(),
                targets=prediction_dfs[i][f'{label} target labels'].to_numpy(),
                normalize='pred',
                fmt=''
            )
        for label in crossvalidation.dataset.labels['regression targets']:
            label_path = model_dir_path / label
            label_path.mkdir(parents=True, exist_ok=True)
            save_prediction_error_figure(
                file_path=label_path / 'prediction_error.pdf',
                target_label=label,
                predictions=prediction_dfs[i][f'{label} predicted labels'].to_numpy(),
                targets=prediction_dfs[i][f'{label} target labels'].to_numpy()
            )
        prediction_dfs[i].to_csv(path_or_buf=model_dir_path / 'prediction.csv')
        classes_dfs[i].to_csv(path_or_buf=model_dir_path / 'classes.csv')


def save_crossvalidation(crossvalidation: CrossvalidationNN, dir_path: Path):
    dir_path.mkdir(parents=True, exist_ok=True)
    torchdefault.save(crossvalidation.state_dict(), f=dir_path / 'crossvalidation_state_dict.pt')
    save_crossvalidation_data(crossvalidation, dir_path)

def load_crossvalidation(dir_path: Path):
    cvnn = CrossvalidationNN()
    cvnn.load_state_dict(torchdefault.load(f=dir_path / 'crossvalidation_state_dict.pt'))
    return cvnn

def load_trainer(dir_path: Path):
    trainer = TrainerNN()
    trainer.load_state_dict(torchdefault.load(f=dir_path / 'trainer_state_dict.pt'))
    return trainer

def save_tensor_dataset(tensor_df: DataFrame, dataset_df: DataFrame, indexes_df: DataFrame, dir_path: Path):
    dir_path.mkdir(parents=True,exist_ok=True)
    tensor_df.to_csv(path_or_buf=dir_path / 'tensor.csv', index=False)
    dataset_df.to_csv(path_or_buf=dir_path / 'dataset.csv', index=False, header=False)
    indexes_df.to_csv(path_or_buf=dir_path / 'indices.csv', index=False)

def load_dataset(file_path: Path, **kwargs):
    return CsvDataset.from_state_dict(state_dict=torchdefault.load(
        f=file_path, **kwargs
    ))

def save_trainer(trainer: TrainerNN, dir_path: Path):
    dir_path.mkdir(parents=True,exist_ok=True)
    torchdefault.save(trainer.states['fit']['state_dict'], f=dir_path / 'fit_state_dict.pt')
    torchdefault.save(trainer.state_dict(), f=dir_path / 'trainer_state_dict.pt')
