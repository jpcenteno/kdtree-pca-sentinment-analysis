#!/usr/bin/env python3

'''
La clase HPOLogger provee
'''

from pathlib import Path
from time import process_time_ns
from typing import List, Any
import csv

class HPOLogger(object):
    '''
    Registra el accuracy para distintas configuraciones de hiperparametros.
    '''
    __slots__ = (
        '_hyperparameter_names',
        '_writer',
        '_i',
        '_f'
    )

    def __init__(self, out_file: Path, hyperparameter_names: List[str]):
        '''
        Inicializa el logger.

        Arguments
        ---------
        hyperparameter_names : List[str]
            Lista de nombres de hiperparámetros.
        '''
        self._hyperparameter_names = hyperparameter_names
        self._i = 0

        fieldnames = ['id'] + hyperparameter_names + ['time', 'acc', 'score']
        self._f = open(out_file, 'w')
        self._writer = csv.DictWriter(self._f, fieldnames=fieldnames)
        self._writer.writeheader()
        self._f.flush()

    def log(self, hyp_vals, acc, time, score):
        self._i += 1
        hyp_dict = dict(zip(self._hyperparameter_names, hyp_vals))
        data_dict = {'id': self._i, 'time': time, 'acc': acc, 'score': score}
        self._writer.writerow({**hyp_dict, **data_dict})
        self._f.flush()
