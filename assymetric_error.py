import numpy as np

def turnover_error(y_true: np.array, y_pred: np.array) -> float:
    '''Takes arrays of predictions and fact values
    and returns assymetric MAE'''
    greater_true = y_true[y_true <= y_pred]
    greater_pred = y_pred[y_true <= y_pred]
    
    less_true = y_true[y_true > y_pred]
    less_pred = y_pred[y_true > y_pred]
    
    error_missed = (1.1*(less_true - less_pred)).sum()
    error_big = (greater_pred - greater_true).sum()
    error = (error_missed + error_big)/len(y_true)
    
    return error