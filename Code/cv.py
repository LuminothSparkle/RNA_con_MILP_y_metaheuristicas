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
        return DataFrame({label : self.binarizers['targets'][label].inverse_transform(data, threshold = 0).squeeze()
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
            
        metric_list = ['accuracy','recall','precision','balanced accuracy','log loss','average precision','roc auc',
                    'matthews corrcoef','f1 score','absolute','absolute percentage','squared','r2 score','d2 absolute', 'explained variance']
        metrics = {metric : {} for metric in metric_list}
        metrics['dataframes'] = {'targets' : targets_dataframe, 'predictions' : predictions_dataframe}
        metrics['targets'] = targets; metrics['predictions'] = predictions
        for label in self.labels['class targets'] :
            metrics['accuracy'][label] = accuracy_score(targets_dataframe[label].to_numpy(),predictions_dataframe[label].to_numpy(), normalize = True)
            metrics['recall'][label] = recall_score(targets_dataframe[label].to_numpy(), predictions_dataframe[label].to_numpy(), average = 'weighted')
            metrics['precision'][label] = precision_score(targets_dataframe[label].to_numpy(), predictions_dataframe[label].to_numpy(), average = 'weighted')
            metrics['balanced accuracy'][label] = balanced_accuracy_score(targets_dataframe[label].to_numpy(),predictions_dataframe[label].to_numpy())
            metrics['log loss'][label] = log_loss(targets[label], predictions[label])
            metrics['average precision'][label] = average_precision_score(targets[label], predictions[label])
            metrics['roc auc'][label] = roc_auc_score(targets[label], predictions[label])
            metrics['matthews corrcoef'][label] = matthews_corrcoef(targets_dataframe[label].to_numpy(), predictions_dataframe[label].to_numpy())
            metrics['f1 score'][label] = f1_score(targets_dataframe[label].to_numpy(), predictions_dataframe[label].to_numpy(), average = 'weighted')
        for label in self.labels['regression targets'] :
            metrics['absolute'][label] = mean_absolute_error(targets_dataframe[label].to_numpy(), predictions_dataframe[label].to_numpy())
            metrics['absolute percentage'][label] = mean_absolute_percentage_error(targets_dataframe[label].to_numpy(), predictions_dataframe[label].to_numpy())
            metrics['squared'][label] = mean_squared_error(targets_dataframe[label].to_numpy(), predictions_dataframe[label].to_numpy())
            metrics['r2 score'][label] = r2_score(targets_dataframe[label].to_numpy(), predictions_dataframe[label].to_numpy())
            metrics['d2 absolute'][label] = d2_absolute_error_score(targets_dataframe[label].to_numpy(), predictions_dataframe[label].to_numpy())
            metrics['explained variance'][label] = explained_variance_score(targets_dataframe[label].to_numpy(), predictions_dataframe[label].to_numpy())
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
    cv = {**{label : [] for label in ['model', 'optimizer', 'scheduler', 'train loss', 'loss', 'metrics']}, 'dataloader' : { 'test' : [] }}
    try :
        for it,(train_dataset,test_dataset) in zip(range(iterations),dataset.split(crossvalidator=crossvalidator)) :     # type: ignore
            if verbose : 
                print(f'iteracion {it}')
            test_size = len(test_dataset); train_size = len(train_dataset); batch_size = train_size // train_batches
            train_dataloader = DataLoader(dataset = train_dataset, shuffle = True, batch_size = batch_size, drop_last = (train_size % train_batches == 1))
            test_dataloader = DataLoader(dataset = test_dataset, batch_size = test_size)
            cv['model'] += [ LinealNN(C = C, hyperparams = hyperparams) ] # type: ignore
            cv['optimizer'] += [ base_optimizer(cv['model'][-1].parameters(),**opt_kwargs) ] # type: ignore
            if base_scheduler is not None :
                cv['scheduler'] += [base_scheduler(cv['optimizer'][-1] ,**sch_kwargs) ] # type: ignore
            cv['train loss'] += [ cv['model'][-1].train_loop(dataloader = train_dataloader, epochs = epochs,
                                    test_dataloader = test_dataloader, optimizer = cv['optimizer'][-1] , loss_fn = loss_fn, 
                                    scheduler = cv['scheduler'][-1] if base_scheduler is not None else None, verbose = verbose, early_tolerance = early_tolerance) ] # type: ignore
            cv['loss'] += [ cv['train loss'][-1]['best'] ] # type: ignore
            cv['model'][-1].load_state_dict(cv['train loss'][-1]['model dict']) # type: ignore
            cv['dataloader']['test'] += [test_dataloader] # type: ignore
            cv['metrics'] += [dataset.metrics(cv['model'][-1],test_dataloader)] # type: ignore
    except KeyboardInterrupt as KbI :
        print(f'{KbI}')
    return cv

def crossvalidation_metrics(dataset : CrossvalidationDataset, hyperparams : dict[str,Any], metrics : dict[str,Any]) :
    cv = {}
    ns_i = perf_counter_ns()
    cv = crossvalidation(C = hyperparams['C'], dataset = dataset,
                                                base_optimizer = hyperparams['optimizer'], opt_kwargs = hyperparams['optimizer kwds'] if 'optimizer kwds' in hyperparams else {},
                                                loss_fn = hyperparams['loss fn'], epochs = hyperparams['epochs'], iterations = hyperparams['iterations'],
                                                base_scheduler = hyperparams['scheduler'] if 'scheduler' in hyperparams else None, 
                                                sch_kwargs = hyperparams['scheduler kwds'] if 'scheduler kwds' in hyperparams else {}, verbose = True,
                                                crossvalidator = hyperparams['crossvalidator'], train_batches = hyperparams['train batches'], 
                                                early_tolerance = hyperparams['early tolerance'] if 'early tolerance' in hyperparams else None, hyperparams = hyperparams)
    ns_t = perf_counter_ns()
    t_ns = ns_t - ns_i
    ns, mus = t_ns % 1000, t_ns // 1000
    mus,ms = mus % 1000, mus // 1000
    ms, s = ms % 1000, ms // 1000
    s, m = s % 60, s // 60
    m, h = m % 60, m // 60
    h, d = h % 24, h // 24
    print(f'time lapsed : {t_ns:>d} ns')
    print(f'time lapsed : {d:>d} d {h:>2d} h {m:>2d} m {s:>2d} s {ms:>3d} ms {mus:>3d} mus {ns:>3d} ns')
    cv['total population'] = len(dataset) # type: ignore
    cv['best it'] = numpy.argmin(cv['loss']) # type: ignore
    cv['worst it'] = numpy.argmax(cv['loss']) # type: ignore
    cv['mean'] = {}
    cv['best prediction'] = {}
    cv['best target'] = {}
    cv['worst prediction'] = {}
    cv['worst target'] = {}
    cv['all percentage metrics'] = {}
    print(f'Total de instancias : {cv["total population"]}')
    print(f'Validacion cruzada con {repr(hyperparams["crossvalidator"])}')
    print(f'Iteraciones totales : {len(cv["loss"])}')
    for label in dataset.labels['class targets'] :
        cv['mean'][label] = { metric : numpy.mean( [ metric_data[metric][label] for metric_data in cv['metrics'] ] ) for metric in metrics['class'] } # type: ignore
        cv['all percentage metrics'][label] = array([ [ metrics[metric][label] for metrics in cv['metrics'] ] for metric in metrics['class percentages'] ]) # type: ignore
        cv['best prediction'][label] = cv['metrics'][cv['best it']]['dataframes']['predictions'][label] # type: ignore
        cv['best target'][label] = cv['metrics'][cv['best it']]['dataframes']['targets'][label] # type: ignore
        cv['worst prediction'][label] = cv['metrics'][cv['worst it']]['dataframes']['predictions'][label] # type: ignore
        cv['worst target'][label] = cv['metrics'][cv['worst it']]['dataframes']['targets'][label] # type: ignore
        fig,ax = pyplot.subplots(); ax.boxplot(numpy.transpose(cv['all percentage metrics'][label]), tick_labels = metrics['class percentages']) # type: ignore
        ax.set(ylabel = 'Porcentaje', title = f'Metricas {label.capitalize()}')
        for metric in metrics['class'] :
            if metric in metrics['class percentages'] :
                print(f'crossvalidation {label.capitalize():20} {metric:20} : {cv["mean"][label][metric]:>2.3%}') # type: ignore
            else :
                fig, ax = pyplot.subplots(); ax.boxplot(array([ metrics[metric][label] for metrics in cv['metrics'] ]), tick_labels = [metric]) # type: ignore
                ax.set(ylabel = 'Valor', title = f'Metrica {label.capitalize()}')
                print(f'crossvalidation {label.capitalize():20} {metric:20} : {cv["mean"][label][metric]:>.7g}') # type: ignore
        for title,target,prediction in [('peor prediccion',cv['worst target'][label],cv['worst prediction'][label]),('mejor prediccion',cv['best target'][label],cv['best prediction'][label])] :
            for conf_matrix_type,format,sufijo in [('all',f'>2.3%','porcentajes'),(None,f'>3d','totales'),('true',f'>2.3%','sobre verdaderos'),('pred',f'>2.3%','sobre predichos')] :
                cm_disp = ConfusionMatrixDisplay.from_predictions(y_true = target, y_pred = prediction, # type: ignore
                                                            labels = dataset.encoders['targets'][label].classes_, normalize = conf_matrix_type, values_format = f'{format}') # type: ignore
                cm_disp.ax_.set(title = f'Matriz de confusion de la {title} de {label.capitalize()}, {sufijo}', xlabel = f'{label.capitalize()} predicho', ylabel = f'{label.capitalize()} real')
    for label in dataset.labels['regression targets'] : # type: ignore
        cv['mean'][label] = { metric : numpy.mean( [ metric_data[metric][label] for metric_data in cv['metrics'] ] ) for metric in metrics['regression'] } # type: ignore
        cv['all percentage metrics'][label] = array([ [ metrics[metric][label] for metrics in cv['metrics'] ] for metric in metrics['regression percentages'] ]) # type: ignore
        cv['best prediction'][label] = cv['metrics'][cv['best it']]['dataframes']['predictions'][label] # type: ignore
        cv['best target'][label] = cv['metrics'][cv['best it']]['dataframes']['targets'][label] # type: ignore
        cv['worst prediction'][label] = cv['metrics'][cv['worst it']]['dataframes']['predictions'][label] # type: ignore
        cv['worst target'][label] = cv['metrics'][cv['worst it']]['dataframes']['targets'][label] # type: ignore
        fig, ax = pyplot.subplots(); ax.boxplot(numpy.transpose(cv['all percentage metrics'][label]), tick_labels = metrics['regression percentages']) # type: ignore
        ax.set(ylabel = 'Porcentaje', title = f'Metricas {label.capitalize()}')
        for metric in metrics['regression'] :
            if metric in metrics['regression percentages'] :
                print(f'crossvalidation {label.capitalize():20} {metric:20} : {cv["mean"][label][metric]:>2.3%}') # type: ignore
            else :
                fig, ax = pyplot.subplots(); ax.boxplot(array([ metrics[metric][label] for metrics in cv['metrics'] ]), tick_labels = [metric]) # type: ignore
                ax.set(ylabel = 'Valor', title = f'Metrica {label.capitalize()}')
                print(f'crossvalidation {label.capitalize():20} {metric:20} : {cv["mean"][label][metric]:>.7g}') # type: ignore
        for title,target,prediction in [('peor regresion',cv['worst target'][label],cv['worst prediction'][label]),('mejor regresion',cv['best target'][label],cv['best prediction'][label])] :
            cm_disp = PredictionErrorDisplay.from_predictions(y_true = target, y_pred = prediction, kind = 'actual_vs_predicted') # type: ignore
            cm_disp.ax_.set(title = f'Regresion de la {title} de {label.capitalize()}', xlabel = f'{label.capitalize()} predicho', ylabel = f'{label.capitalize()} real')
    pyplot.show()