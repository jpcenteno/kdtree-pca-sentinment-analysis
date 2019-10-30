'''
Este modulo contiene funciones relacionadas con la experimentación sobre el
tamaño muestral y vocabulario.
'''

import pandas as pd

import csv

import itertools
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score

from time import process_time

import sentiment

# ----------------------------------------------------------------------------
# Parametros
# ----------------------------------------------------------------------------


SUBSAMPLING_RATIOS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                      0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

HPARAMS = [(10, 100), (50, 100), (150, 100), (100, 10), (100, 20), (100, 50)]
N_REPEATS = 10

# ----------------------------------------------------------------------------
# Data:
# ----------------------------------------------------------------------------


def read_data():
    '''
    Lee los datos del disco y los vectoriza.
    '''

    # Read data:
    df = pd.read_csv("data/imdb_small.csv", index_col=0)
    text_train = df[df.type == 'train']["review"]
    label_train = df[df.type == 'train']["label"]
    text_test = df[df.type == 'test']["review"]
    label_test = df[df.type == 'test']["label"]

    # Vectorizo los datos. Se usan estos max_df y min_df.
    vectorizer = CountVectorizer(max_df=0.90, min_df=0.01, max_features=5000)
    vectorizer.fit(text_train)
    X_train = vectorizer.transform(text_train)
    y_train = (label_train == 'pos').values
    X_test = vectorizer.transform(text_test)
    y_test = (label_test == 'pos').values

    return X_train, y_train, X_test, y_test


def subsample(X_train, y_train, ratio):
    '''
    Realiza un subsampling del set de entrenamiento. Respeta un 50/50 entre las
    labels.
    '''
    print(f'N_train original = {X_train.shape[0]}')
    sss = StratifiedShuffleSplit(n_splits=1, test_size=(1 - ratio))
    subsample_index, _ = next(sss.split(X_train, y_train))
    X_train_subsample = X_train[subsample_index]
    y_train_subsample = y_train[subsample_index]
    print(f'N_train subsample = {X_train_subsample.shape[0]}')
    return X_train_subsample, y_train_subsample


# ----------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------

def _repeat(it, n):
    return itertools.chain.from_iterable(itertools.repeat(it, n))


def get_max_alpha(hparams):
    '''
    Devuelve el maximo alpha entre los hiperparametros.
    '''
    return max([alpha for _, alpha in hparams])

def get_acc(k, X_train, y_train, X_test, y_test):
    # Construye clasificador KNN
    knn = sentiment.KNNClassifier(k)
    knn.fit(X_train, y_train)
    # Obtiene accuracy:
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

if __name__ == '__main__':

    X_train, y_train, X_test, y_test = read_data()
    alpha_max = get_max_alpha(HPARAMS)


    with open('exp-vocabulario.csv', 'w') as f:

        csv = csv.writer(f)
        csv.writerow(['n', 'k', 'alpha', 'acc'])

        for ratio in _repeat(SUBSAMPLING_RATIOS, N_REPEATS):

            print('Ratio:', ratio)

            # Saco un subsample para ese ratio:
            X_train_subsample, y_train_subsample = subsample(X_train, y_train,
                                                             ratio)

            # Obtiene tamaño sub-set de entrenamiento
            n = X_train_subsample.shape[0]

            # Aplico PCA con el alpha mas grande. Esto es para hacerlo una sola
            # vez. despues puedo simplificar reduciendo las columnas.
            pca = sentiment.PCA(alpha_max)
            pca.fit(X_train_subsample)
            X_train_subsample_pca = pca.transform(X_train_subsample)
            X_test_pca = pca.transform(X_test)

            for k, alpha in HPARAMS:

                print(f'calculando para k, alpha = {k}, {alpha}')

                # Usa el [:, :alpha] para aprovechar un único PCA reduciendo la
                # cantidadde columnas.
                acc = get_acc(k=k,
                              X_train=X_train_subsample_pca[:, :alpha],
                              y_train=y_train_subsample,
                              X_test=X_test_pca[:, :alpha],
                              y_test=y_test)

                print(f'{n},{k},{alpha},{acc}')
                csv.writerow([n, k, alpha, acc])
                f.flush()
