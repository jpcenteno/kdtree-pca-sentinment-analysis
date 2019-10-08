#!/usr/bin/env python
from simpleai.search import SearchProblem
from simpleai.search.local import hill_climbing

from sklearn.metrics import accuracy_score
# es el PCA que soporta data rala ¯\_(ヅ)_/¯
#from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA

from time import process_time

# lo ideal es que esto sea parámetro, pero bue
from sklearn.neighbors import KNeighborsClassifier
import sys
sys.path.append('../build/')
import sentiment

import random

######################################################################
###                                                                ###
###      Algunos valores por defecto, para la clase y el main      ###
###                                                                ###
######################################################################
dataset_default="../data/imdb_small.csv"
initial_neightbours_default=100
neightbours_step_default=10
print_log_default=True
usar_pca_default=True

##################################################################################
###                                                                            ###
### La definición del problema para que lo resuelva algun algoritmo de simpleai###
###                                                                            ###
##################################################################################

class KNNHyperParameters(SearchProblem):

    def __init__(self, X_train, Y_train, X_test, Y_test
                 ,classifier_from="sentiment", pca_from="sentiment"
                 ,min_time=5, max_time=15, time_penalization=1.2
                 ,neightbours_step = neightbours_step_default, initial_neightbours=initial_neightbours_default
                 ,usar_pca=usar_pca_default, initial_pca = None, print_log=print_log_default):

        """Recibe conjuntos de entreamiento y testeo y dos strings
        classifier_from y pca_from, que pueden ser sentiment si se usa
        la librería en C++, o sklearn, si se usa ese framework de
        python.

        initial_neightbours es la cantidad e vecinos iniciales, si no se especifica toma 100
        initial_pca es la cantidad de componentes principales si es None, toma el 5% del vocabulario de X_train

        min_time y max_time son minutos, min_time es el tiempo por debajo
        del cual, solo nos importa la precision. Entre ambos valores, tomamos una razón entre precisión y tiempo.
        time_penalization es el factor por el que se multiplica el tiempo, en el score_ratio, si este supera max_time; dandole más peso al incremento de tiempo (y bajando el score)

        """

        if initial_pca == None:
            initial_pca = int(X_train.shape[1] / 20)

        self.neightbours_step = neightbours_step
        self.pca_step = neightbours_step
        super().__init__((initial_neightbours, initial_pca))

        if classifier_from == "sklearn":
            self.classifier_klass_constructor = KNeighborsClassifier
        else:
            self.classifier_klass_constructor = sentiment.KNNClassifier

        if pca_from == "sklearn":
            self.pca_klass_constructor = lambda a: PCA(n_components=a)
        else:
            self.pca_klass_constructor = lambda a: sentiment.PCA(a)

        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

        self.min_time = min_time
        self.max_time = max_time
        self.time_penalization = time_penalization

        self.usar_pca = usar_pca
        self.print_log = print_log

    def actions(self, state):
        """this method receives a state, and must return the list of actions that can be performed from that particular state.

        La idea es:
        """
        k, alfa = state
        nexts = [(n, a)
                 for n in [-self.neightbours_step, +self.neightbours_step]
                 for a in [-self.pca_step, +self.pca_step]]
        self.log("Estoy en {}, considerando frontera...".format(state))
        return nexts

    def result(self, state, action):
        """this method receives a state and an action, and must return the resulting state of applying that particular action from that particular state"""
        k, alfa = state
        neigh_step, pca_step = action
        return (k + neigh_step, alfa + pca_step)

    def value(self, state):
        """This method receives a state, and returns a valuation (“score”) of that value. Better states must have higher scores."""
        self.log("Calculando Score de {}".format(state))
        k, alfa = state

        beg = process_time()
        if self.usar_pca:
            self.log("Transformando datos (PCA)")
            time_log = process_time()
            pca = self.pca_klass_constructor(alfa)
            x_train = pca.fit_transform(self.X_train)
            x_test = pca.fit_transform(self.X_test)
            self.log("listo - elapsed {} segundos en PCA".format(process_time() - time_log))
        else:
            x_train = self.X_train
            x_test = self.X_test

        self.log("Fitteando y Prediciendo")
        time_log = process_time()
        clf = self.classifier_klass_constructor(k)
        clf.fit(x_train, self.Y_train)
        y_pred = clf.predict(x_test)
        end = process_time()
        self.log("listo - elapsed {} segundos en KNN".format(end - time_log))
        self.log("tiempo total: {}".format(end - beg))

        acc = accuracy_score(self.Y_test, y_pred)
        time = (end - beg) / 60.0

        score = self._score(time, acc)
        self.log("Evaluando: {} => Accuracy: {}, Time: {} minutos, Score: {}".format(state, acc, time, score))
        return score

    def _score(self, time, acc):
        """agregada para poder sobreescribirla en una clase hija de ser necesario"""
        if time < self.min_time:
            return acc / self.min_time  # divido por min_time para desempatar los valores del intervalo
        elif time < self.max_time:
            return acc / time
        return acc / (self.time_penalization * time) # acá tengo problemas con empates, pero fue

    def generate_random_state(self):
        """this method receives nothing, and must return a randomly generated state.

        Devuelvo un valor entre las dos dimesiones de X_train dividido
        por 10. Muestras de entrenamiento / 10 => max vecinos,
        vocabulario / 10 => max alfa"""

        return (random.randrange(1, int(self.X_train.shape[0] / 10)), random.randrange(1, int(self.X_train.shape[1] / 10)))

    def log(self, msg):
        if self.print_log:
            print(msg)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Hacer alguna busqueda local sobre los hiperparámetros de KNN.')
    parser.add_argument('implementation', choices=["sentiment", "sklearn"]
                        ,help='usar "sentiment" nuestra implementación de KNN y PCA o la de la biblioteca "sklearn"')
    parser.add_argument('-n', type=int, default=initial_neightbours_default
                        ,help='La cantidad inicial de vecinos a considerar - por defecto usa el de la clase')
    parser.add_argument('--alpha', type=int, default=None
                        ,help='La cantidad de componentes principales incial a considerar - por defecto usa el de la clase')
    parser.add_argument('--print-log', type=bool, default=print_log_default
                        ,help='Si imprime los logs a medida de que avanza - por defecto usa el de la clase')
    parser.add_argument('--n-step', type=int, default=neightbours_step_default
                        ,help='El tamaño del paso al moverse por el vecindario en la dimensión de vecinos - por defecto usa el de la clase')
    parser.add_argument('--use-pca', dest='use_pca', action='store_true'
                        ,help='Indica que se usará PCA')
    parser.add_argument('--not-use-pca', dest='use_pca', action='store_false'
                        ,help='Indica que NO se usará PCA')
    parser.add_argument('--data-set', type=str, default=dataset_default
                        ,help='path del dataset, puede ser relativo descomprimido - por defecto usa ../../data/imdb_small.csv')
    parser.add_argument('--algorithm', choices=["hill_climbing", "beam"], default="hill_climbing"
                        ,help='El algoritmo a usar para la búsqueda')
    parser.add_argument('--beam-size', type=int, default=100
                        ,help='Si se usa beamer, la cantidad de estados iniciales que se considera - por defecto 100')
    parser.add_argument('--iterations_limit', type=int, default=None
                        ,help="Si se pasa, acota la cantidad de iteraciones - por defecto sigue hasta que no puede mejorar")
    parser.add_argument('--use-sparse-override', dest='use_sparse_override', action='store_true'
                        ,help='Le pasa matrices ralas a las funciones de knn y pca siempre')
    parser.add_argument('--use-dense-override', dest='use_sparse_override', action='store_false'
                        ,help='Le pasa matrices densas a las funciones de knn y pca siempre')
    parser.set_defaults(use_pca=usar_pca_default,use_sparse_override=None)

    args = parser.parse_args()

    # BEGIN CHORIPASTEO
    import pandas as pd

    #!cd ../../data && tar -xvf *.tgz
    #!cd ../../data && tar -xvf *.tar.gz

    df = pd.read_csv(args.data_set, index_col=0)

    print("Cantidad de documentos: {}".format(df.shape[0]))

    text_train = df[df.type == 'train']["review"]
    label_train = df[df.type == 'train']["label"]

    text_test = df[df.type == 'test']["review"]
    label_test = df[df.type == 'test']["label"]

    print("Class balance : {} pos {} neg".format(
    (label_train == 'pos').sum() / label_train.shape[0],
    (label_train == 'neg').sum() / label_train.shape[0]))
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(max_df=0.90, min_df=0.01, max_features=5000)

    vectorizer.fit(text_train)
    # ENDCHORIPASTEO

    print(args)
    if args.use_sparse_override == True:
        print("Alimentando PCA y KNN con matrices ralas forzozamente")
        X_train, y_train = vectorizer.transform(text_train), (label_train == 'pos').values
        X_test, y_test = vectorizer.transform(text_test), (label_test == 'pos').values
    elif args.use_sparse_override == False:
        print("Alimentando PCA y KNN con matrices densas forzozamente")
        X_train, y_train = vectorizer.transform(text_train).todense(), (label_train == 'pos').values
        X_test, y_test = vectorizer.transform(text_test).todense(), (label_test == 'pos').values
    else:
        if args.use_pca and args.implementation=='sklearn':
            print("Usando sklearn con PCA, transformando matrices a densas desde python.")
            X_train, y_train = vectorizer.transform(text_train).todense(), (label_train == 'pos').values
            X_test, y_test = vectorizer.transform(text_test).todense(), (label_test == 'pos').values
        else:
            print("Usando sentiment o sklearn sin PCA, dejamos las matrices esparsas")
            X_train, y_train = vectorizer.transform(text_train), (label_train == 'pos').values
            X_test, y_test = vectorizer.transform(text_test), (label_test == 'pos').values

    print("Creando Problema")
    knn_problem = KNNHyperParameters(X_train, y_train, X_test, y_test
                                     ,classifier_from=args.implementation, pca_from=args.implementation
                                     ,neightbours_step=args.n_step, initial_neightbours=args.n, initial_pca=args.alpha
                                     ,usar_pca=args.use_pca, print_log=args.print_log)

    from simpleai.search.viewers import BaseViewer
    visor = BaseViewer()

    if args.algorithm == "hill_climbing":
        print("Resolviendo con Hill Climbing")
        from simpleai.search.local import hill_climbing
        result = hill_climbing(knn_problem, viewer=visor, iterations_limit=args.iterations_limit)
        print("Encontramos: {}\nLuego de este camino: {}\n".format(result.state, result.path()))
    if args.algorithm == "beam":
        print("Resolviendo con Beam")
        from simpleai.search.local import beam
        result = beam(knn_problem, viewer=visor, beam_size=args.beam_size, iterations_limit=args.iterations_limit)
        print("Encontramos: {}\nLuego de este camino: {}\n".format(result.state, result.path()))
