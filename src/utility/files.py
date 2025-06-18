"""
A
"""
from typing import Any
from pathlib import Path
import json
import pickle
from pickle import Pickler, Unpickler
import torch
from torch.optim import (
    Adadelta, Adafactor, Adagrad, Adam,
    Adamax, AdamW, ASGD, SGD, SparseAdam
)
from torch.optim.lr_scheduler import (
    LinearLR, SequentialLR, StepLR,
    CyclicLR, OneCycleLR, ConstantLR
)
from torch.nn.init import calculate_gain
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
from sklearn.model_selection import (
    RepeatedStratifiedKFold, RepeatedKFold, KFold,
    StratifiedGroupKFold, StratifiedKFold, GroupKFold
)
from src.utility.nn.lineal import LinealNN
from src.utility.nn.crossvalidation import CrossvalidationDataset

type_dict = {
    'RepeatedStratifiedKFold' : RepeatedStratifiedKFold,
    'RepeatedKFold' : RepeatedKFold,
    'KFold' : KFold,
    'StratifiedGroupKFold' : StratifiedGroupKFold,
    'StratifiedKFold' : StratifiedKFold,
    'GroupKFold' : GroupKFold,
    'ones' : torch.ones,
    'zeros' : torch.zeros,
    'rand' : torch.rand,
    'randint' : torch.randint,
    'LinearLR' : LinearLR,
    'SequentialLR' : SequentialLR,
    'StepLR' : StepLR,
    'CyclicLR' : CyclicLR,
    'OneCycleLR' : OneCycleLR,
    'ConstantLR' : ConstantLR,
    'Adadelta' : Adadelta,
    'Adafactor' : Adafactor,
    'Adagrad' : Adagrad,
    'Adam' : Adam,
    'Adamax' : Adamax,
    'AdamW' : AdamW,
    'ASGD' : ASGD,
    'SGD' : SGD,
    'SparseAdam' : SparseAdam,
    'calculate_gain' : calculate_gain
}

def save_module(module : LinealNN, dir_path : Path, name : str = '', exists_ok : bool = True) :
    """
    A
    """
    assert exists_ok or all(
        len(sorted(dir_path.glob(f'{name}*'))) == 0
    ), 'Alguno de los archivos ya existe'
    save_object(module, dir_path / f'{name}.pkl')
    torch.save(module, dir_path / f'{name}.pt')
    onnx_program = torch.onnx.export(
        model = module,
        args = (
            torch.ones(
                (1,module.capacity[0]),
                dtype = torch.double
            ),
        ),
        dynamo = True,
        export_params = True
    )
    onnx_program.optimize()
    onnx_program.save( destination = dir_path / f'{name}.onnx')
    traced = torch.jit.trace(module, (torch.rand(1,module.capacity[0])))
    torch.jit.script(traced).save(dir_path / f'{name}_script.pt')

def safe_suffix(name : str, suffix : str) :
    """
    A
    """
    if not name.endswith('_') and name != '' :
        return f'{name}_{suffix}'
    return f'{name}{suffix}'

def save_models(models : list[LinealNN], dir_path : Path, name : str = '', exists_ok = True) :
    """
    A
    """
    dir_path.mkdir(parents = True, exist_ok = True)
    name_dict = {
        safe_suffix(name,f'{i}') : models[i]
        for i in range(len(models))
    }
    name_dict[safe_suffix(name,'best')]  = models[0]
    name_dict[safe_suffix(name,'worst')] = models[-1]
    suffixes = ['.pt','.pkl','.onnx', 'script.pt']
    assert exists_ok or all(
        not (dir_path / safe_suffix(name,suffix)).exists()
        for suffix in suffixes
        for name in name_dict
    ), 'Alguno de los archivos ya existe'
    for name, model in name_dict.items() :
        save_module(model, dir_path, name)

def save_object(obj : Any, file_path : Path) :
    """
    A
    """
    with file_path.open('wb') as fo :
        pickle.dump(obj,fo,5)

def read_object(file_path : Path) :
    """
    A
    """
    with file_path.open('rb') as fo :
        obj = pickle.load(fo)
    return obj

def write_mdb(file_path : Path, module : LinealNN, dataset : CrossvalidationDataset) :
    """
    A
    """
    with file_path.open('wb') as fo :
        pickler = Pickler(fo,5)
        pickler.dump(tuple(module,dataset))

def read_mdb(file_path : Path) :
    """
    A
    """
    with file_path.open('rb') as fo :
        unpickler = Unpickler(fo)
        module, dataset = unpickler.load()
    return module, dataset

def save_mdbs(
    modules : list[LinealNN], datasets : list[CrossvalidationDataset],
    dir_path : Path, name : str = '', exists_ok : bool = True
) :
    """
    A
    """
    assert len(modules) == len(datasets), 'El numero de elementos no coinciden'
    dir_path.mkdir(parents = True, exist_ok = True)
    name_dict = {
        safe_suffix(name,f'{i}') : (modules[i],datasets[i])
        for i in range(len(modules))
    }
    name_dict[safe_suffix(name,'best')]  = (modules[0],  modules[0])
    name_dict[safe_suffix(name,'worst')] = (modules[-1], modules[-1])
    assert exists_ok or all(
        not (dir_path / f'{safe_suffix(name,"mdb")}.pkl').exists()
        for name in name_dict
    ), 'Alguno de los archivos ya existe'
    for name, (model, dataset) in name_dict.items() :
        write_mdb(dir_path / f'{safe_suffix(name,"mdb")}.pkl', model, dataset)

def save_boxplot_figure(file_path : Path, target_label : str,
    values : dict[str,ndarray], percentage : bool = False
) :
    """
    A
    """
    fig, ax = pyplot.subplots()
    ax.boxplot(
        numpy.concat( [ numpy.atleast_2d(data).T for data in values.values() ], axis = 1 ),
        tick_labels = [ label.capitalize() for label in values ]
    )
    ax.set(
        ylabel = 'Porcentaje' if percentage else 'Valor',
        title = f'Metricas de {target_label.capitalize()}'
    )
    fig.savefig(file_path, transparent = True)
    return fig

def save_confusion_matrix_figure(
    file_path : Path, target_label : str, classes : list[str],
    targets : ndarray, predictions : ndarray,
    fmt : str | None = None, normalize : str | None = None
) :
    """
    A
    """
    cm_disp = ConfusionMatrixDisplay.from_predictions(
        y_true = targets, y_pred = predictions,
        labels = classes,
        normalize = normalize,
        values_format = fmt if fmt is not None else '>d' if normalize is None else '>2.3%'
    )
    cm_disp.ax_.set(
        title = f'Matriz de confusion de {target_label.capitalize()}{
            f", normalizado sobre {normalize}" if normalize is not None else ''
        }',
        xlabel = f'{target_label.capitalize()} predicho',
        ylabel = f'{target_label.capitalize()} real'
    )
    cm_disp.figure_.savefig(file_path, transparent = True)
    return cm_disp

def save_prediction_error_figure(
    file_path : Path, target_label : str, targets : ndarray, predictions : ndarray,
) :
    """
    A
    """
    pe_disp = PredictionErrorDisplay.from_predictions(
        y_true = targets, y_pred = predictions, kind = 'actual_vs_predicted'
    )
    pe_disp.ax_.set(
        title = f'Regresion de {target_label.capitalize()}',
        xlabel = f'{target_label.capitalize()} predicho',
        ylabel = f'{target_label.capitalize()} real'
    )
    pe_disp.figure_.savefig(file_path, transparent = True)
    return pe_disp

def save_class_target_metrics(
    file_path : Path, target_label : str,
    data : list[tuple[ndarray,ndarray,ndarray,ndarray]]
) :
    """
    A
    """
    metrics_dataframes = []
    for tensor_targets, scores, targets, predictions in data :
        model_metrics = {}
        model_metrics['accuracy'] = [accuracy_score(
            y_true = targets,
            y_pred = predictions,
            normalize = True
        )]
        model_metrics['recall'] = [recall_score(
            y_true = targets,
            y_pred = predictions,
            average = 'weighted'
        )]
        model_metrics['precision'] = [precision_score(
            y_true = targets,
            y_pred = predictions,
            average = 'weighted'
        )]
        model_metrics['f1 score'] = [f1_score(
            y_true = targets,
            y_pred = predictions,
            average = 'weighted'
        )]
        model_metrics['balanced accuracy'] = [balanced_accuracy_score(
            y_true = targets,
            y_pred = predictions
        )]
        model_metrics['log loss'] = [log_loss(
            y_true = tensor_targets,
            y_pred = scores
        )]
        model_metrics['average precision'] = [average_precision_score(
            y_true = tensor_targets,
            y_score = scores
        )]
        model_metrics['roc auc'] = [roc_auc_score(
            y_true = tensor_targets,
            y_score = scores
        )]
        model_metrics['matthews corrcoef'] = [matthews_corrcoef(
            y_true = targets,
            y_pred = predictions
        )]
        metrics_dataframes += [DataFrame(model_metrics)]
    metrics = pandas.concat(metrics_dataframes, axis = 'index', ignore_index = True)
    metrics.to_csv(
        file_path,
        index_label = f'Metricas de clasificación para {target_label}',
        encoding = 'utf-8',
        header = True,
        index = True
    )
    return metrics

def save_crossvalidation(
    dir_path : Path, cv_data : dict,
    name : str = '', exists_ok : bool = True
) :
    """
    A
    """
    dir_path.mkdir(parents = True, exist_ok = True)
    class_data, regression_data = {}, {}
    model_predictions = []
    for model,dataset,test_dataloader in zip(
        cv_data['model'],cv_data['dataset'],cv_data['test dataloader']
    ) :
        predictions = dataset.prediction(model,test_dataloader)
        model_predictions += [predictions]
        for label in dataset.labels['class targets'] :
            if label not in class_data :
                class_data[label] = []
            class_data[label] += [predictions[label]]
        for label in dataset.labels['regression targets'] :
            if label not in regression_data :
                regression_data[label] = []
            regression_data[label] += [predictions[label]]
    save_models(cv_data['model'],dir_path = dir_path / 'models', name = name, exists_ok = exists_ok)
    save_mdbs(
        cv_data['model'], cv_data['dataset'], dir_path = dir_path / 'mdbs',
        name = name, exists_ok = exists_ok
    )
    for name_type, i in {'best' : 0, 'worst' : -1}.items() :
        save_displays(
            model_predictions = model_predictions[i],
            class_encoders = cv_data['dataset'][i].encoders,
            class_targets = cv_data['dataset'][i].labels['class targets'],
            dir_path = dir_path / name_type,
            name = name,
            exists_ok = exists_ok
        )
    boxplot_figures_path = dir_path / 'boxplots'
    boxplot_figures_path.mkdir(parents = True, exist_ok = True)
    class_labels_data, regression_labels_data, *_ = save_gen_metrics(
        class_data = class_data,
        regression_data = regression_data,
        dir_path = boxplot_figures_path,
        name = name
    )
    dataframes_path = dir_path / 'dataframes'
    dataframes_path.mkdir(parents = True, exist_ok = True)
    save_dataframes(
        model_data = {'time lapsed' : cv_data['train time'], 'loss' : cv_data['loss'] },
        class_labels_data = class_labels_data,
        regression_labels_data = regression_labels_data,
        dir_path = dataframes_path,
        name = name,
        exists_ok = exists_ok
    )
    log_path = dir_path / f'{name}.log'
    save_log(
        class_labels_data = class_labels_data,
        regression_labels_data = regression_labels_data,
        total_nanoseconds = numpy.sum(cv_data['train time']),
        log_path = log_path
    )

def save_dataframes(
    model_data : dict,
    class_labels_data : dict,
    regression_labels_data : dict,
    dir_path : Path,
    name : str = '',
    exists_ok : bool = True
) :
    """
    A
    """
    dir_path.mkdir(parents = True, exist_ok = True)
    assert exists_ok or all([
        not (dir_path / f'{safe_suffix(name,suffix)}.csv').exists()
        for suffix in ['model','class','regression']
    ])
    DataFrame(model_data).to_csv(
        dir_path / f'{safe_suffix(name,"model")}.csv',
        index_label = 'Metricas para cada modelo',
        index = True,
        header = True,
        encoding = 'utf-8'
    )
    for labels_data, labels_metrics_path in [
        (class_labels_data, dir_path / f'{safe_suffix(name,"class")}.csv',),
        (regression_labels_data, dir_path / f'{safe_suffix(name,"regression")}.csv',)
    ] :
        DataFrame(labels_data).to_csv(
            labels_metrics_path,
            index_label = 'Metricas para cada etiqueta',
            index = True,
            header = True,
            encoding = 'utf-8'
        )

def save_displays(
    model_predictions : dict,
    class_targets : list[str],
    class_encoders : dict,
    dir_path : Path,
    name : str = '',
    extensions : list[str] | None = None,
    exists_ok : bool = True
) :
    """
    A
    """
    dir_path.mkdir(parents = True, exist_ok = True)
    if extensions is None :
        extensions = ['eps','png','pdf','svg','tiff','ps','raw']
    formats = {
        None : '>d', 'all' : '>2.3%',
        'true' : '>2.3%', 'pred' : '>2.3%'
    }
    assert exists_ok or all([
        *[
            not (dir_path / f'{safe_suffix(name,label)}_pe.{extension}').exists()
            for label in model_predictions
            for extension in extensions
        ],
        *[
            not (dir_path / f'{safe_suffix(name,label)}_cm_{norm}.{extension}').exists()
            for label in model_predictions
            for extension in extensions
            for norm in formats
        ]
    ])
    for label,data in model_predictions.items() :
        if label in class_targets :
            _, _, targets, pred = data
            classes = class_encoders[label].classes_
            base_name = f'{safe_suffix(name,label)}_cm'
            for norm,fmt in formats.items() :
                for extension in extensions :
                    save_confusion_matrix_figure(
                        dir_path / f'{base_name}_{norm}.{extension}',
                        label, classes, targets, pred, fmt, norm
                    )
                    pyplot.close()
        else :
            targets, pred = data
            for extension in extensions :
                save_prediction_error_figure(
                    dir_path / f'{safe_suffix(name,label)}_pe.{extension}',
                    label, targets, pred
                )
                pyplot.close()

def save_log(
    class_labels_data : dict,
    regression_labels_data : dict,
    total_nanoseconds : int,
    log_path : Path
) :
    """
    A
    """
    ns, mus = total_nanoseconds % 1000, total_nanoseconds // 1000
    mus, ms = mus % 1000, mus // 1000
    ms, s = ms % 1000, ms // 1000
    s, m = s % 60, s // 60
    m, h = m % 60, m // 60
    h, d = h % 24, h // 24
    with log_path.open('wt', encoding = 'utf-8') as fp :
        fp.write(
f'''
Total time lapsed   {d} days {h} hours {m} minutes {s} seconds
                    {ms} miliseconds {mus} microseconds {ns} nanoseconds
'''
        )
        for labels_data in [class_labels_data, regression_labels_data] :
            for label,metrics in labels_data.items() :
                for metric_name, value in metrics.items() :
                    fp.write(f'Mean metric {metric_name} for {label} : {value}\n')

def save_gen_metrics(
    class_data : dict, regression_data :dict,
    dir_path : Path, name : str = '',
    percentage_metrics : list[str] | None = None,
    extensions : list[str] | None = None,
    exists_ok : bool = True
) :
    """
    A
    """
    if percentage_metrics is None :
        percentage_metrics = [
            'roc auc','precision','accuracy','recall','f1 score',
            'balanced accuracy','average precision','balanced accuracy',
            'matthews corrcoef'
        ]
    labels_data = []
    for data_type, data_manager in [
        (class_data,      save_class_target_metrics),
        (regression_data, save_regression_target_metrics)
    ] :
        labels_data_type = {}
        for label, data in data_type.items() :
            data_metrics = data_manager(
                dir_path / f'{safe_suffix(name,label)}.csv',
                label, data
            )
            save_boxplots(
                dir_path = dir_path, name = name,
                label = label, metrics = data_metrics,
                percentage_metrics = percentage_metrics,
                extensions = extensions,
                exists_ok = exists_ok
            )
            labels_data_type[label] = data_metrics.mean(axis = 'index')
        labels_data += [labels_data_type]
    return tuple(labels_data)

def save_boxplots(
    dir_path : Path,
    name : str,
    label : str,
    metrics : DataFrame,
    percentage_metrics : list[str] | None = None,
    extensions : list[str] | None = None,
    exists_ok : bool = True
) :
    """
    A
    """
    dir_path.mkdir(parents = True, exist_ok = True)
    if extensions is None :
        extensions = ['eps','png','pdf','svg','tiff','ps','raw']
    if percentage_metrics is None :
        percentage_metrics = []
    for metric in metrics.columns :
        if metric not in percentage_metrics :
            for extension in extensions :
                file_path = dir_path / f'{safe_suffix(name,label)}_{metric}.{extension}'
                assert exists_ok or not file_path.exists(), f'El archivo {file_path} ya existe'
                save_boxplot_figure(
                    file_path,
                    label,
                    { metric : metrics[metric].to_numpy() },
                    False
                )
                pyplot.close()
        for extension in extensions :
            file_path = dir_path / f'{safe_suffix(name,label)}_percentages.{extension}'
            assert exists_ok or not file_path.exists(), f'El archivo {file_path} ya existe'
            save_boxplot_figure(
                file_path,
                label,
                {
                    metric : metrics[metric].to_numpy()
                    for metric in percentage_metrics
                },
                True
            )
            pyplot.close()

def save_regression_target_metrics(
    file_path : Path, target_label : str,
    data : list[tuple[ndarray,ndarray]]
) :
    """
    A
    """
    metrics_dataframes = []
    for targets, predictions in data :
        model_metrics = {}
        model_metrics['absolute'] = mean_absolute_error(
            y_true = targets,
            y_pred = predictions
        )
        model_metrics['absolute percentage'] = mean_absolute_percentage_error(
            y_true = targets,
            y_pred = predictions
        )
        model_metrics['squared'] = mean_squared_error(
            y_true = targets,
            y_pred = predictions
        )
        model_metrics['r2 score'] = r2_score(
            y_true = targets,
            y_pred = predictions
        )
        model_metrics['d2 absolute'] = d2_absolute_error_score(
            y_true = targets,
            y_pred = predictions
        )
        model_metrics['explained variance'] = explained_variance_score(
            y_true = targets,
            y_pred = predictions
        )
        metrics_dataframes += [DataFrame(model_metrics)]
    metrics = pandas.concat(metrics_dataframes, axis = 'index', ignore_index = True)
    metrics.to_csv(
        file_path, index_label = f'Metricas de regresion para {target_label}',
        encoding = 'utf-8', header = True, index = True
    )
    return metrics

def analize_value_param(value : Any) -> tuple | Any :
    """
    A
    """
    if isinstance(value,dict) and 'kwds' in value :
        if value['name'] in type_dict :
            return type_dict[value['name']], analize_value_param(value['kwds'])
        else :
            return value['name'], analize_value_param(value['kwds'])
    elif isinstance(value,list) :
        return [ analize_value_param(value) for value in value ]
    elif isinstance(value,dict) :
        return { name : analize_value_param(value) for name,value in value.items() }
    elif isinstance(value,str) and value in type_dict :
        return type_dict[value]
    else :
        return value

def analize_layers_param(value : Any, layers : int) -> dict[int,Any] :
    """
    A
    """
    if isinstance(value,list) :
        return { k : analize_value_param(value) for k,value in enumerate(value) }
    elif isinstance(value,dict) and not 'kwds' in value :
        return { int(k) : analize_value_param(value) for k, value in value.items() }
    else :
        return { k : analize_value_param(value) for k in range(layers) }

def read_arch_json(file_path : Path) -> dict :
    """
    A
    """
    arch = {}
    with file_path.open('rt', encoding = 'utf-8') as fp :
        raw = json.load(fp)
        arch['layers'] = len(raw['capacity']) + 1
        for param, values in raw.items() :
            if param == 'layers params' :
                for param, values in values.items() :
                    arch[param] = analize_layers_param(values,arch['layers'])
            else :
                arch[param] = analize_value_param(values)
    return arch

def read_cv_json(file_path : Path) -> dict :
    """
    A
    """
    cv = {}
    with file_path.open('rt', encoding = 'utf-8') as fp :
        raw = json.load(fp)
        for param, values in raw.items() :
            cv[param] = analize_value_param(values)
    return cv
