'''
Este modulo contiene funciones relacionadas con la experimentación sobre el
tamaño muestral y vocabulario.
'''

import itertools
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score

from knnpca import PCAKneighboursClasifier

from time import process_time

def exp_train_subsample(X_train, y_train, X_test, y_test, subsampling_ratio,
                        k=100, alpha=100):
    '''
    Computa el accuracy bajo un subsample del conjunto de entrenamiento.

    Parameters
    ----------
    X_train : Matriz (n, m)
        Training data.
    y_train : Vector (n,)
        Vector con los target value para `X_train`.
    X_test : Matriz (n, m)
        Training data.
    y_test : Vector (n,)
        Vector con los target value para `X_test`.
    subsampling_ratio : float en rango (0, 1]
        Ratio para subsamplear X_train.
    k : int en rango {1 ... n}, optional
        Hyperparameter de KNN.
    alpha : int en rango {1 ... m}, optional
        Hyperparameter de PCA.

    Returns
    -------
    acc : float
        Accuracy obtenido para KNN con PCA subsampleando `(X_train, y_train)`
        con el ratio `subsampling_ratio`.
    n_train : int > 0
        Tamaño muestral del set de entrenamiento.
    k : int en rango {1 ... n}, optional
        Hyperparameter de KNN.
    alpha : int en rango {1 ... m}, optional
        Hyperparameter de PCA.
    time_fit : float
        Tiempo de cpu en segundos para el fit.
    time_predict : float
        Tiempo de cpu en segundos para el predict.
    '''

    # Subsampling
    sss = StratifiedShuffleSplit(n_splits=1, test_size=subsampling_ratio)
    subsample_index, _ = next(sss.split(X_train, y_train))
    X_train, y_train = X_train[subsample_index], y_train[subsample_index]

    time_start = process_time()
    pcaknn = PCAKneighboursClasifier(k, alpha)
    time_end = process_time()
    time_fit = time_end - time_start

    time_start = process_time()
    pcaknn.fit(X_train, y_train)
    time_end = process_time()
    time_predict = time_end - time_start

    y_pred = pcaknn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return {'n_train': len(subsample_index),
            'k': k,
            'alpha': alpha,
            'acc': acc,
            'time_fit': time_fit,
            'time_predict': time_predict}


def exp_grid_train_subsample(X_train, y_train, X_test, y_test,
                             subsampling_ratios, ks, alphas):
    '''
    Computa el accuracy bajo un subsample del conjunto de entrenamiento.

    Parameters
    ----------
    X_train : Matriz (n, m)
        Training data.
    y_train : Vector (n,)
        Vector con los target value para `X_train`.
    X_test : Matriz (n, m)
        Training data.
    y_test : Vector (n,)
        Vector con los target value para `X_test`.
    subsampling_ratios : List[float] en rango (0, 1]
        Ratio para subsamplear X_train.
    ks : List[int] en rango {1 ... n}, optional
        Hyperparameter de KNN.
    alphas : List[int] en rango {1 ... m}, optional
        Hyperparameter de PCA.

    Returns
    -------
    results: List[Dict] de:
        acc : float
            Accuracy obtenido para KNN con PCA subsampleando
            `(X_train, y_train)` con el ratio `subsampling_ratio`.
        n_train : int > 0
            Tamaño muestral del set de entrenamiento.
        k : int en rango {1 ... n}, optional
            Hyperparameter de KNN.
        alpha : int en rango {1 ... m}, optional
            Hyperparameter de PCA.
        time_fit : float
            Tiempo de cpu en segundos para el fit.
        time_predict : float
            Tiempo de cpu en segundos para el predict.
    '''
    for ratio, k, alpha in itertools.product(subsampling_ratios, ks, alphas):
        print(f'ratio: {ratio}, k: {k}, alpha: {alpha}')
        try:
            result = exp_train_subsample(X_train, y_train, X_test, y_test,
                                         ratio, k=k, alpha=alpha)
            yield(result)
        except ValueError: # YOLO
            print('Raised valueerror')

