import torch
from torch.nn import Module
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch.nn.functional import cross_entropy, softmax, mse_loss, sigmoid, binary_cross_entropy_with_logits

from collections.abc import Iterable, Callable, Generator

from sklearn.model_selection import BaseCrossValidator
from sklearn.metrics import (ConfusionMatrixDisplay, PredictionErrorDisplay,
                             accuracy_score, mean_absolute_percentage_error, mean_absolute_error,
                             recall_score, roc_auc_score, accuracy_score, precision_score, average_precision_score,
                             balanced_accuracy_score, mean_squared_error, r2_score, d2_absolute_error_score,
                             matthews_corrcoef, f1_score, log_loss, explained_variance_score)

from scipy.sparse import spmatrix

import numpy
from numpy import array
from numpy.typing import ArrayLike

from sklearn.model_selection import BaseCrossValidator
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler

from pandas import DataFrame, Index

from typing import Any
from abc import abstractmethod

import matplotlib.pyplot as pyplot

from time import perf_counter_ns

from lnn import LinealNN

class CrossvalidationDataset(Dataset) :
    labels : dict[str,Any]
    @abstractmethod
    def split(self, crossvalidator : BaseCrossValidator) -> Generator:
        pass
    @abstractmethod
    def encode(self, target : DataFrame) -> Tensor :
        pass
    @abstractmethod
    def decode(self, pred : Tensor) -> DataFrame:
        pass
    @abstractmethod
    def metrics(self, model : Module, dataloader : DataLoader | None = None) -> dict:
        pass
    @abstractmethod
    def loss_fn(self, pred : Tensor, target : Tensor) -> Tensor :
        pass
    
class CrossvalidationTensorDataset(CrossvalidationDataset) :
    tensors : dict[str,Tensor | dict[dict,Tensor]]
    dataset : TensorDataset
    binarized : dict[str,dict[str,ArrayLike | spmatrix]]
    encoded : dict[str,dict[str,ArrayLike | spmatrix]]
    labels : dict[str,Index]
    binarizers : dict[str,dict[str,LabelBinarizer]]
    encoders : dict[str,dict[str,LabelEncoder]]
    scalers : dict[str,dict[str,StandardScaler] | StandardScaler]
    scaled : dict[str,dict[str,ArrayLike | spmatrix] | ArrayLike | spmatrix]
    dataframes : dict[str,DataFrame]
    tensor_size : dict[str,dict[str,int]]
    
    def __init__(self, dataframe : DataFrame, labels : dict[str,Index], scale : bool = False) :
        self.dataframes = {}
        self.tensors = {'class targets' : {}, 'regression targets' : {}, 'weights' : {}}
        self.tensor_size = {'targets' : {}, 'features' : {}}
        self.binarizers = {'targets' : {}, 'features' : {}}
        self.binarized = {'targets' : {}, 'features' : {}}
        self.encoders = {'targets' : {}, 'features' : {}}
        self.encoded = {'targets' : {}, 'features' : {}}
        self.scalers = {'targets' : {}, 'features' : {}}
        self.scaled = {'targets' : {}, 'features' : {}}
        self.dataframes['all'] = dataframe
        self.labels = labels
        for dataframe_type in ['class targets','features','regression targets'] :
            self.dataframes[dataframe_type] = self.dataframes['all'][self.labels[dataframe_type]]
        self.scalers['features'] = StandardScaler()
        self.scaled['features'] = self.scalers['features'].fit_transform(self.dataframes['features'])
        for label in self.labels['class targets'] :
            self.binarizers['targets'][label] = LabelBinarizer()
            self.binarized['targets'][label] = self.binarizers['targets'][label].fit_transform(self.dataframes['class targets'][label])
            self.encoders['targets'][label] = LabelEncoder()
            self.encoded['targets'][label] = self.encoders['targets'][label].fit_transform(self.dataframes['class targets'][label])
            self.tensors['class targets'][label] = torch.tensor(data = self.binarized['targets'][label], dtype = torch.double)
            self.tensors['weights'][label] = (1 - self.tensors['class targets'][label].mean(dim = 0)) * (len(self.encoders['targets'][label].classes_) / (len(self.encoders['targets'][label].classes_) - 1))
        for label in self.labels['regression targets'] :
            self.tensors['regression targets'][label] = torch.tensor(data = self.dataframes['regression targets'][label], dtype = torch.double).unsqueeze(dim = 1)
        self.tensors['targets'] = torch.column_stack( [*[tensor for tensor in self.tensors['class targets'].values()],*[tensor for tensor in self.tensors['regression targets'].values()]] )
        if scale :
            self.tensors['features'] = torch.tensor( data = self.scaled['features'], dtype = torch.double )
        else :
            self.tensors['features'] = torch.tensor( data = self.dataframes['features'].to_numpy(), dtype = torch.double )
        self.tensor_size['targets'] = {label : tensor.size(dim = 1) for label,tensor in [*self.tensors['class targets'].items(),*self.tensors['regression targets'].items()] } # type: ignore
        self.dataset = TensorDataset(self.tensors['features'], self.tensors['targets'])
    
    def size_targets(self) -> int :
        return self.tensors['targets'].size(dim = 1) # type: ignore

    def size_features(self) -> int :
        return self.tensors['features'].size(dim = 1) # type: ignore

    def __len__(self) -> int :
        return len(self.dataset)
    
    def __getitem__(self, idx) -> tuple[Tensor, Tensor] :
        with torch.device('cuda') :
            features, targets = self.dataset[idx]
        return features, targets
    
    def encode(self, target : DataFrame)  -> Tensor:
        return torch.column_stack([
                *[torch.tensor(self.binarizers['targets'][label].transform(target[label].to_numpy()), dtype = torch.double) for label in self.labels['class targets']],
                torch.tensor(target[self.labels['regression targets']].to_numpy(), dtype = torch.double)
                ])

    def split_pred_tensor(self, pred : Tensor) :
        pred_dict, = tuple( dict(zip(label_tuple, pred.split(size_tuple, dim = 1)))
                          for label_tuple, size_tuple in [ tuple( zip( *self.tensor_size['targets'].items() ) ) ] )
        return pred_dict
    
    def decode(self, pred : Tensor)  -> DataFrame:
        return DataFrame({label : self.binarizers['targets'][label].inverse_transform(data, threshold = 0).squeeze() # type: ignore
                                       if label in self.binarizers['targets'] else data.squeeze()
                                       for label_list, tensor_list in [ tuple( zip( *self.split_pred_tensor(pred).items() ) ) ]
                                       for label, data in zip(label_list, ( tensor.cpu().numpy() for tensor in tensor_list ))
                                       })
    
    def split(self, crossvalidator : BaseCrossValidator) :
        yield from ( (Subset(self,train_index.tolist()), Subset(self,test_index.tolist()))
                    for train_index, test_index in crossvalidator.split(self.dataframes['features'].to_numpy(), self.dataframes['class targets'].to_numpy())
                   )
    
    def metrics(self, model : Module, dataloader : DataLoader | None = None) -> dict :
        with torch.inference_mode() :
            if dataloader is None :
                dataloader = DataLoader(dataset = self, batch_size = len(self))
            targets, predictions = ( torch.vstack(tensors) for tensors in zip(*( (y, model(X)) for X,y in dataloader )) )
            targets_dataframe, predictions_dataframe = ( self.decode(tensor) for tensor in (targets, predictions) )    
            targets, predictions = ( self.split_pred_tensor(tensor) for tensor in (targets, predictions) )
            predictions = { label : tensor if label in self.labels['regression targets'] else sigmoid(tensor) if tensor.size(dim = 1) == 1 else softmax(tensor) for label,tensor in predictions.items() }
            targets, predictions = ({ label : tensor.squeeze().cpu().numpy() for label,tensor in tensor_dict.items() } for tensor_dict in (targets, predictions))
        metrics = {}
        metrics['dataframes'] = { 'targets' : targets_dataframe, 'predictions' : predictions_dataframe }
        metrics['tensors'] = { 'targets' : targets, 'predictions' : predictions }
        for label in self.labels['class targets'] :
            metrics[label] = {}
            metrics[label]['targets'] = targets_dataframe[label].to_numpy()
            metrics[label]['predictions'] = predictions_dataframe[label].to_numpy()
            metrics[label]['targets tensor'] = targets[label]
            metrics[label]['predictions tensor'] = predictions[label]
            metrics[label]['accuracy'] = accuracy_score(targets_dataframe[label].to_numpy(),predictions_dataframe[label].to_numpy(), normalize = True)
            metrics[label]['recall'] = recall_score(targets_dataframe[label].to_numpy(), predictions_dataframe[label].to_numpy(), average = 'weighted')
            metrics[label]['precision'] = precision_score(targets_dataframe[label].to_numpy(), predictions_dataframe[label].to_numpy(), average = 'weighted')
            metrics[label]['balanced accuracy'] = balanced_accuracy_score(targets_dataframe[label].to_numpy(),predictions_dataframe[label].to_numpy())
            metrics[label]['log loss'] = log_loss(targets[label], predictions[label])
            metrics[label]['average precision'] = average_precision_score(targets[label], predictions[label])
            metrics[label]['roc auc'] = roc_auc_score(targets[label], predictions[label])
            metrics[label]['matthews corrcoef'] = matthews_corrcoef(targets_dataframe[label].to_numpy(), predictions_dataframe[label].to_numpy())
            metrics[label]['f1 score'] = f1_score(targets_dataframe[label].to_numpy(), predictions_dataframe[label].to_numpy(), average = 'weighted')
        for label in self.labels['regression targets'] :
            metrics[label] = {}
            metrics[label]['targets'] = targets_dataframe[label].to_numpy()
            metrics[label]['predictions'] = predictions_dataframe[label].to_numpy()
            metrics[label]['targets tensor'] = targets[label]
            metrics[label]['predictions tensor'] = predictions[label]
            metrics[label]['absolute'] = mean_absolute_error(targets_dataframe[label].to_numpy(), predictions_dataframe[label].to_numpy())
            metrics[label]['absolute percentage'] = mean_absolute_percentage_error(targets_dataframe[label].to_numpy(), predictions_dataframe[label].to_numpy())
            metrics[label]['squared'] = mean_squared_error(targets_dataframe[label].to_numpy(), predictions_dataframe[label].to_numpy())
            metrics[label]['r2 score'] = r2_score(targets_dataframe[label].to_numpy(), predictions_dataframe[label].to_numpy())
            metrics[label]['d2 absolute'] = d2_absolute_error_score(targets_dataframe[label].to_numpy(), predictions_dataframe[label].to_numpy())
            metrics[label]['explained variance'] = explained_variance_score(targets_dataframe[label].to_numpy(), predictions_dataframe[label].to_numpy())
        return metrics
    
    def loss_fn(self, pred : Tensor, target : Tensor) -> Tensor :
        target_splitted = self.split_pred_tensor(target)
        pred_splitted = self.split_pred_tensor(pred)
        loss = torch.tensor(0, device = 'cuda', dtype = torch.double)
        list = [*zip(pred_splitted.items(),target_splitted.values())]
        for (label,pred),target in list :
            if label in self.labels['regression targets'] :
                loss += mse_loss(pred,target, reduction = 'mean')
            elif pred.size(dim = 1) > 1 : 
                loss += cross_entropy(pred,target, reduction = 'mean', weight = self.tensors['weights'][label]) # type: ignore
            else :
                loss += binary_cross_entropy_with_logits(pred,target, reduction = 'mean', pos_weight = self.tensors['weights'][label]) # type: ignore
        return loss / len(list)        

def crossvalidation(dataset : CrossvalidationDataset, base_optimizer : type[Optimizer], loss_fn : Callable[[Tensor,Tensor],Tensor], C : Iterable[int], 
                    epochs : int, iterations : int, crossvalidator : BaseCrossValidator, train_batches : int = 10, 
                    opt_kwargs : dict[str,Any] = {}, base_scheduler : type[LRScheduler] | None = None, sch_kwargs : dict[str,Any] = {},
                    verbose : bool = False, early_tolerance : int | None = None, hyperparams : dict[str,Any] = {}) :
    label_list = ['metrics','loss','train time','train loss','scheduler','model','optimizer','test dataset','train dataset','test size','train size']
    cv = {**{label : [] for label in label_list}}
    try :
        for it,(train_dataset,test_dataset) in zip(range(iterations),dataset.split(crossvalidator=crossvalidator)) :     # type: ignore
            if verbose : 
                print(f'iteracion {it}')
            test_size = len(test_dataset); train_size = len(train_dataset); batch_size = train_size // train_batches
            train_dataloader = DataLoader(dataset = train_dataset, shuffle = True, batch_size = batch_size, drop_last = (train_size % train_batches == 1))
            test_dataloader = DataLoader(dataset = test_dataset, batch_size = test_size)
            model = LinealNN(C = C, hyperparams = hyperparams)
            optimizer = base_optimizer(model.parameters(),**opt_kwargs)
            if base_scheduler is not None :
                scheduler = base_scheduler(optimizer ,**sch_kwargs)
            else :
                scheduler = None
            ns_i = perf_counter_ns()
            train_loss = model.train_loop(dataloader = train_dataloader, epochs = epochs,
                                    test_dataloader = test_dataloader, optimizer = optimizer , loss_fn = loss_fn, 
                                    scheduler = scheduler, verbose = verbose, early_tolerance = early_tolerance)
            ns_t = perf_counter_ns()
            model.load_state_dict(train_loss['model dict']) # type: ignore
            cv['test size'] += [test_size]; cv['train size'] += [train_size] # type: ignore
            cv['optimizer'] += [optimizer]; cv['scheduler'] += [scheduler]  # type: ignore
            cv['train loss'] += [ train_loss ] # type: ignore
            cv['model'] += [ model ] # type: ignore
            cv['train time'] += [ns_t - ns_i]  # type: ignore
            cv['loss'] += [ train_loss['best'] ] # type: ignore
            cv['test dataset'] += [test_dataset] # type: ignore
            cv['train dataset'] += [train_dataset] # type: ignore
            cv['metrics'] += [dataset.metrics(model,test_dataloader)] # type: ignore
    except KeyboardInterrupt as KbI :
        print(f'{KbI}')
    sorted = numpy.argsort(array(cv['loss'])) # type: ignore
    for label in label_list :
        cv[label] = [ cv[label][it] for it in sorted ]
    cv['metric list'] = {} # type: ignore
    for label in (label for label in cv['metrics'][0] if label not in ['tensors','dataframes']) :
        cv['metric list'][label] = { metric_label : array( [ metric_data[label][metric_label] for metric_data in cv['metrics'] ] ) for metric_label in cv['metrics'][0][label] if metric_label not in ['targets','predictions','predictions tensor','targets tensor'] } # type: ignore
    cv['crossvalidator'] = hyperparams['crossvalidator']
    cv['encoders'] = dataset.encoders # type: ignore
    cv['labels'] = dataset.labels # type: ignore
    cv['iterations'] = len(cv['loss']) # type: ignore
    cv['total size'] = len(dataset) # type: ignore
    cv['total time'] = numpy.sum(array(cv['train time']))
    return cv

def visualize_results(cv : dict[str,Any], metrics : dict[str,Any]) :
    t_ns = cv['total time']
    ns, mus = t_ns % 1000, t_ns // 1000 # type: ignore
    mus,ms = mus % 1000, mus // 1000
    ms, s = ms % 1000, ms // 1000
    s, m = s % 60, s // 60
    m, h = m % 60, m // 60
    h, d = h % 24, h // 24
    print(f'time lapsed : {t_ns:>d} ns')
    print(f'time lapsed : {d:>d} d {h:>2d} h {m:>2d} m {s:>2d} s {ms:>3d} ms {mus:>3d} mus {ns:>3d} ns')
    print(f'Total de instancias : {cv["total size"]}')
    print(f'Validacion cruzada con {repr(cv["crossvalidator"])}')
    print(f'Iteraciones totales : {cv["iterations"]}')
    for label in cv['labels']['class targets'] :
        mean = { metric_label : numpy.mean( cv['metric list'][label][metric_label] ) for metric_label in metrics['class'] }
        percentage_metrics = array( [ cv['metric list'][label][metric_label] for metric_label in metrics['class percentages'] ] )
        fig,ax = pyplot.subplots(); ax.boxplot(numpy.transpose(percentage_metrics), tick_labels = metrics['class percentages']) # type: ignore
        ax.set(ylabel = 'Porcentaje', title = f'Métricas {label.capitalize()}')
        for metric in metrics['class'] :
            if metric in metrics['class percentages'] :
                print(f'crossvalidation {label.capitalize():20} {metric:20} : {mean[metric]:>2.3%}') # type: ignore
            else :
                fig, ax = pyplot.subplots(); ax.boxplot(cv['metric list'][label][metric], tick_labels = [metric]) # type: ignore
                ax.set(ylabel = 'Valor', title = f'Metrica {label.capitalize()}')
                print(f'crossvalidation {label.capitalize():20} {metric:20} : {mean[metric]:>.7g}') # type: ignore
        for title,target,prediction in [('peor prediccion',cv['metrics'][-1][label]['targets'],cv['metrics'][-1][label]['predictions']),('mejor prediccion',cv['metrics'][0][label]['targets'],cv['metrics'][0][label]['predictions'])] :
            for conf_matrix_type,format,sufijo in [('all',f'>2.3%','porcentajes'),(None,f'>3d','totales'),('true',f'>2.3%','sobre verdaderos'),('pred',f'>2.3%','sobre predichos')] :
                cm_disp = ConfusionMatrixDisplay.from_predictions(y_true = target, y_pred = prediction, # type: ignore
                                                            labels = cv['encoders']['targets'][label].classes_, normalize = conf_matrix_type, values_format = f'{format}') # type: ignore
                cm_disp.ax_.set(title = f'Matriz de confusion de la {title} de {label.capitalize()}, {sufijo}', xlabel = f'{label.capitalize()} predicho', ylabel = f'{label.capitalize()} real')
    for label in cv['labels']['regression targets'] : # type: ignore
        mean = { metric_label : numpy.mean( cv['metric list'][label][metric_label] ) for metric_label in metrics['regression'] }
        percentage_metrics = array( [ cv['metric list'][label][metric_label] for metric_label in metrics['regression percentages'] ] )
        fig, ax = pyplot.subplots(); ax.boxplot(numpy.transpose(percentage_metrics), tick_labels = metrics['regression percentages']) # type: ignore
        ax.set(ylabel = 'Porcentaje', title = f'Métricas {label.capitalize()}')
        for metric in metrics['regression'] :
            if metric in metrics['regression percentages'] :
                print(f'crossvalidation {label.capitalize():20} {metric:20} : {mean[metric]:>2.3%}') # type: ignore
            else :
                fig, ax = pyplot.subplots(); ax.boxplot(cv['metric list'][label][metric], tick_labels = [metric]) # type: ignore
                ax.set(ylabel = 'Valor', title = f'Metrica {label.capitalize()}')
                print(f'crossvalidation {label.capitalize():20} {metric:20} : {mean[metric]:>.7g}') # type: ignore
        for title,target,prediction in [('peor regresion',cv['metrics'][-1][label]['targets'],cv['metrics'][-1][label]['predictions']),('mejor regresion',cv['metrics'][0][label]['targets'],cv['metrics'][0][label]['predictions'])] :
            cm_disp = PredictionErrorDisplay.from_predictions(y_true = target, y_pred = prediction, kind = 'actual_vs_predicted') # type: ignore
            cm_disp.ax_.set(title = f'Regresion de la {title} de {label.capitalize()}', xlabel = f'{label.capitalize()} predicho', ylabel = f'{label.capitalize()} real')
    pyplot.show()