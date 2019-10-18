#! /usr/bin/env python3
import logging
import sys
from time import process_time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

sys.path.append("notebooks/")
from sentiment import KNNClassifier

# ------------------------------------------------------------------------------
# Configuración del script
# ------------------------------------------------------------------------------

K = 100

# Dejar en 0 para medir con el dataset completo.
N_test = 0

logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')

# ------------------------------------------------------------------------------
# Carga de datos:
# ------------------------------------------------------------------------------

logging.info('Cargando datos')
df = pd.read_csv("data/imdb_small.csv", index_col=0)

# Separa en train y test
logging.debug(f'Cant. de documentos: {df.shape[0]}')
text_train = df[df.type == 'train']["review"]
label_train = df[df.type == 'train']["label"]
text_test = df[df.type == 'test']["review"]
label_test = df[df.type == 'test']["label"]

# Tira algunas estadisticas
logging.debug(f'Cant. de instancias de entrenamiento: {len(text_train)}')
logging.debug(f'Cant. de instancias de test: {len(text_test)}')
logging.debug('Balance en entrenamiento: {:.2f} pos {} neg'.format(
    100 * ( (label_train == 'pos').sum() / len(label_train) ),
    100 * ( (label_train == 'neg').sum() / len(label_train) ),
))

# ------------------------------------------------------------------------------
# Vectorizado
# ------------------------------------------------------------------------------

logging.info('Vectorizando los datos')
vectorizer = CountVectorizer(max_df=0.90, min_df=0.01, max_features=5000)
vectorizer.fit(text_train)
X_train, y_train = vectorizer.transform(text_train), (label_train == 'pos').values
X_test, y_test = vectorizer.transform(text_test), (label_test == 'pos').values
#  X_train = X_train.todense()
#  X_test = X_test.todense()

# ------------------------------------------------------------------------------
# Entrenamiento
# ------------------------------------------------------------------------------

logging.info(f'Entranando el clasificador. (K={K})')
time_start = process_time()
clf = KNNClassifier(K)
clf.fit(X_train, y_train)
time_finish = process_time()
logging.info(f'Clasificador entrenado en {time_finish - time_start:.4f}s')

# ------------------------------------------------------------------------------
# Midiendo predicción
# ------------------------------------------------------------------------------

if N_test == 0:
    N_test = X_test.shape[0]

logging.info(f'Midiendo tiempos para {N_test} elementos de testing.')
time_start = process_time()
clf.predict(X_test[:N_test])
time_finish = process_time()
logging.info(f'Predict tomó {time_finish - time_start:.4f}s')
