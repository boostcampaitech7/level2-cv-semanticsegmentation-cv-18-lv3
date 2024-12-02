import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_squared_error
import torch
from tqdm import tqdm
from torch import Tensor

class Metric:
    @staticmethod
    def calculate(outputs: Tensor, labels: Tensor):
        raise NotImplementedError

    @staticmethod
    def is_better(new: float, old: float, min_delta: float):
        raise NotImplementedError

class Accuracy(Metric):
    @staticmethod
    def calculate(outputs: Tensor, labels: Tensor) -> float:
        predicted = np.argmax(outputs, axis=1)
        return accuracy_score(labels, predicted)

    worst_value = 0.0

    @staticmethod
    def is_better(new: float, old: float, min_delta: float):
        return new > old + min_delta

class F1Score(Metric):
    @staticmethod
    def calculate(outputs: Tensor, labels: Tensor) -> float:
        predicted = np.argmax(outputs, axis=1)
        return f1_score(labels, predicted, average='weighted')

    worst_value = 0.0

    @staticmethod
    def is_better(new: float, old: float, min_delta: float) -> bool:
        return new > old + min_delta

class Precision(Metric):
    @staticmethod
    def calculate(outputs: Tensor, labels: Tensor) -> float:
        predicted = np.argmax(outputs, axis=1)
        return precision_score(labels, predicted, average='weighted')

    worst_value = 0.0

    @staticmethod
    def is_better(new: float, old: float, min_delta: float) -> bool:
        return new > old + min_delta

class Recall(Metric):
    @staticmethod
    def calculate(outputs: Tensor, labels: Tensor) -> float:
        predicted = np.argmax(outputs, axis=1)
        return recall_score(labels, predicted, average='weighted')

    worst_value = 0.0

    @staticmethod
    def is_better(new: float, old: float, min_delta: float) -> bool:
        return new > old + min_delta

class MSE(Metric):
    @staticmethod
    def calculate(outputs: Tensor, labels: Tensor) -> float:
        return mean_squared_error(labels, outputs)

    worst_value = float('inf')

    @staticmethod
    def is_better(new: float, old: float, min_delta: float) -> bool:
        return new < old - min_delta

class DiceCoef(Metric):
    @staticmethod
    def calculate(outputs: Tensor, labels: Tensor) -> float:
        outputs_f = outputs.flatten(2)
        labels_f = labels.flatten(2)
        intersection = torch.sum(outputs_f * labels_f, -1)
        
        eps = 0.0001
        return (2. * intersection + eps) / (torch.sum(outputs_f, -1) + torch.sum(labels_f, -1) + eps)
    
    worst_value = 0.0 

    @staticmethod
    def is_better(new: float, old: float, min_delta: float) -> bool:
        return new > old + min_delta

def get_metric_function(metric_name: str) -> Metric:
    metrics = {
        'accuracy': Accuracy(),
        'f1': F1Score(),
        'precision': Precision(),
        'recall': Recall(),
        'mse': MSE(),
        'dice_coef': DiceCoef()
    }
    
    if metric_name in metrics:
        return metrics[metric_name]
    else:
        raise ValueError(f"Unknown metric: {metric_name}")