#! /usr/bin/env python3

'''
La idea es mostrar para un par de valores `alpha`, `k`, como varía el
_accuracy_ variando los hiperparametros `max_df`, `min_df`. La **hipotesis** es
que no mejora el accuracy, pues PCA se encarga de eliminar la información
innecesaria.
'''

import time
import logging
import itertools
import csv

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

import sentiment

# ----------------------------------------------------------------------------
# Parámetros para este script.
# ----------------------------------------------------------------------------

# Variar las combinaciones de hiperparametros acá:
ALPHA = 500
K = 50
max_dfs = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
min_dfs = [0.001, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15]

# Configuración del logging
logging.basicConfig(level=logging.WARNING, format='[%(levelname)s] %(message)s')
log = logging.getLogger('notebook')
log.setLevel(logging.DEBUG)

# ----------------------------------------------------------------------------
# Funciones De procesado de datos:
# ----------------------------------------------------------------------------

def read_data(path):
    '''
    Carga el dataset en memoria.
    '''

    log.debug('Cargando datos en memoria.')

    df = pd.read_csv(path, index_col=0)
    log.info("Cantidad de documentos: {}".format(df.shape[0]))

    # Separa los textos en instancias de test y entrenamiento.
    text_train = df[df.type == 'train']["review"]
    label_train = df[df.type == 'train']["label"]
    text_test = df[df.type == 'test']["review"]
    label_test = df[df.type == 'test']["label"]
    log.info(f"Cantidad de instancias de train = {len(text_train)}")
    log.info(f"Cantidad de instancias de test = {len(text_test)}")

    # Calculo el class balance
    pct_pos = (label_train == 'pos').sum() / label_train.shape[0]
    pct_neg = (label_train == 'neg').sum() / label_train.shape[0]
    log.info("Class balance : %.3f pos %.3f neg", pct_pos, pct_neg)

    text_train = df[df.type == 'train']["review"]
    label_train = df[df.type == 'train']["label"]
    text_test = df[df.type == 'test']["review"]
    label_test = df[df.type == 'test']["label"]

    text_data = text_train, label_train, text_test, label_test
    return text_data

def vectorizer(text_data, max_df, min_df):
    '''
    Vectorizador para los datos. En este script es importante.
    '''

    text_train, label_train, text_test, label_test = text_data

    log.info('Vectorizando los datos. Para max_df=%.2f, min_df=%.2f',
              max_df, min_df)
    vectorizer = CountVectorizer(max_df=max_df, min_df=min_df,
                                 max_features=5000)
    vectorizer.fit(text_train)
    X_train, y_train = vectorizer.transform(text_train), (label_train == 'pos').values
    X_test, y_test = vectorizer.transform(text_test), (label_test == 'pos').values

    log.debug(f'X_train.shape = {X_train.shape}')
    log.debug(f'y_train.shape = {y_train.shape}')
    log.debug(f'X_test.shape = {X_test.shape}')
    log.debug(f'y_test.shape = {y_test.shape}')

    return X_train, y_train, X_test, y_test

# ----------------------------------------------------------------------------
# 
# ----------------------------------------------------------------------------

def benchmark(text_data, max_df, min_df):

    time_begin = time.perf_counter()

    log.info('benchmark: (max_df, min_df) = (%.2f, %.2f)', max_df, min_df)
    X_train, y_train, X_test, y_test = vectorizer(text_data, max_df, min_df)

    log.debug(f'fit_transform de PCA.')
    pca = sentiment.PCA(ALPHA, 1e-5)
    X_train_pca = pca.fit_transform(X_train)

    log.debug('Entrenando el clasificador KNN')
    knn_clf = sentiment.KNNClassifier(K)
    knn_clf.fit(X_train_pca, y_train)

    log.debug('Clasificando...')
    X_test_pca = pca.transform(X_test)
    y_pred = knn_clf.predict(X_test_pca)

    log.debug('Calculando accuracy...')
    acc = accuracy_score(y_pred, y_test)
    log.info('Se obtuvo un accuracy de %s', acc)

    time_end = time.perf_counter()
    time_delta = time_end - time_begin
    log.info('Se tardó %.2fs', time_delta)

    return acc, time_delta


if __name__ == '__main__':

    path_to_data = './data/imdb_small.csv'
    text_data = read_data(path_to_data)

    with open('resultados-opti-max-df-min-df.csv', 'w') as f:
        fieldnames = ['max_df', 'min_df', 'acc', 'time']
        writer = csv.DictWriter(f, fieldnames)

        writer.writeheader()
        f.flush()

        for max_df, min_df in itertools.product(max_dfs, min_dfs):
            acc, time_delta = benchmark(text_data, max_df, min_df)
            writer.writerow({'max_df': max_df,
                             'min_df': min_df,
                             'acc': acc,
                             'time': time_delta
                            })
            f.flush()
