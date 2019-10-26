#!/usr/bin/env python
from pathlib import Path
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
import math

def pca_klass_constructor(impl, pca_eps):
    if impl == "sklearn":
        return lambda a: PCA(n_components=a, tol=pca_eps)
    else:
        return lambda a: sentiment.PCA(a, pca_eps)

from hpologger import HPOLogger

######################################################################
###                                                                ###
###      Algunos valores por defecto, para la clase y el main      ###
###                                                                ###
######################################################################
dataset_train_default=Path("../data/imdb_small.csv")
dataset_test_default=Path("../data/test_sample.csv")
dataset_true_default=Path("../data/test_sample.true")
initial_neightbours_default=100
neightbours_step_default=2
pca_step_default=2
pca_eps_default=1e-6
# ver KNNGridDecorator para entender que es divition_scale
divition_scale_default=16
print_log_default=True
usar_pca_default=True
#
beam_size_default=10
##################################################################################
###                                                                            ###
### La definición del problema para que lo resuelva algun algoritmo de simpleai###
###                                                                            ###
##################################################################################

class KNNHyperParameters(SearchProblem):

    __slots__ = (
        "initial_pca",
        "neightbours_step",
        "pca_step",
        "classifier_klass_constructor",
        "pca_klass_constructor",
        "X_train",
        "Y_train",
        "X_test",
        "Y_test",
        "min_time",
        "max_time",
        "time_penalization",
        "usar_pca",
        "pca_eps",
        "memoize_pca",
        "memoize_clf",
        "memoize_state",
        "print_log",
        "metadata",
        "logger"
    )

    def __init__(self, X_train, Y_train, X_test, Y_test
                 ,classifier_from="sentiment", pca_from="sentiment"
                 ,min_time=5, max_time=15, time_penalization=1.2
                 ,neightbours_step = neightbours_step_default
                 ,pca_step = pca_step_default
                 ,initial_neightbours=initial_neightbours_default
                 ,initial_pca = None, pca_eps = pca_eps_default
                 ,usar_pca=usar_pca_default, memoize_pca = {}, print_log=print_log_default
                 ,logger=None):

        """Recibe conjuntos de entrenamiento y testeo y dos strings
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
        self.pca_step = pca_step
        super().__init__((initial_neightbours, initial_pca))

        if classifier_from == "sklearn":
            self.classifier_klass_constructor = KNeighborsClassifier
        else:
            self.classifier_klass_constructor = sentiment.KNNClassifier

        self.pca_klass_constructor = pca_klass_constructor(pca_from, pca_eps)

        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

        self.min_time = min_time
        self.max_time = max_time
        self.time_penalization = time_penalization

        self.usar_pca = usar_pca
        self.pca_eps = pca_eps
        self.memoize_pca = memoize_pca
        self.memoize_clf = {}
        self.memoize_state = {}
        self.print_log = print_log
        self.metadata = None

        self.logger = logger

    def get_classifier(self, neightbours, alfa, x_train):
        if not self.usar_pca:
            alfa=0
        if self.classifier_klass_constructor == sentiment.KNNClassifier:
            if alfa in self.memoize_clf:
                print("Recuperando classifier de memoria!")
                clf = self.memoize_clf[alfa]
                clf.setNeightbors(neightbours)
                return clf
            print("Construyendo y Fitteando Classificador")
            clf = self.classifier_klass_constructor(neightbours)
            clf.fit(x_train, self.Y_train)
            self.memoize_clf[alfa] = clf
        return clf

    def actions(self, state):
        """this method receives a state, and must return the list of actions that can be performed from that particular state. """
        k, alfa = state
        # generamos los siguientes con cuidado de no devolver valores inválidos
        nexts = filter(lambda t: t[0] + k > 1 and t[1] + alfa > 2,
                       [(n, a)
                        for n in [-self.neightbours_step, +self.neightbours_step]
                        for a in [-self.pca_step, +self.pca_step]])
        self.log("Estoy en {}, considerando frontera...".format(state))
        return nexts

    def result(self, state, action):
        """this method receives a state and an action, and must return the resulting state of applying that particular action from that particular state"""
        k, alfa = state
        neigh_step, pca_step = action
        return (k + neigh_step, alfa + pca_step)

    def get_memoized_pca(self, alfa):
        for (a, data) in self.memoize_pca.items():
            print ("explorando memoizado: ({}, {})".format(a, data))
            if a == alfa:
                return data
            if a > alfa:
                x_train, x_test = data
                return x_train[:,0:alfa], x_test[:,0:alfa]
        return None

    def value(self, state):
        """This method receives a state, and returns a valuation (“score”) of that value. Better states must have higher scores."""
        if state in self.memoize_state:
            return self.memoize_state[state]

        self.log("Calculando Score de {}".format(state))
        k, alfa = state

        beg = process_time()
        if self.usar_pca:
            self.log("Transformando datos (PCA)")
            memo_pca = self.get_memoized_pca(alfa)
            if memo_pca is not None:
                print("Ya calculado, obteniendo resultado memoizado (PCA)")
                x_train, x_test = memo_pca
            else:
                print("Calculando (PCA)")
                time_log = process_time()
                pca = self.pca_klass_constructor(alfa)
                pca.fit(self.X_train)
                x_train = pca.transform(self.X_train)
                x_test = pca.transform(self.X_test)
                self.log("listo - elapsed {} segundos en PCA".format(process_time() - time_log))
                if not self.memoize_pca == None:
                    print("Guardando memoizacion (PCA)")
                    self.memoize_pca[alfa] = (x_train, x_test)
        else:
            x_train = self.X_train
            x_test = self.X_test

        self.log("Fitteando y Prediciendo")
        time_log = process_time()
        clf = self.get_classifier(k, alfa, x_train)
        y_pred = clf.predict(x_test)
        end = process_time()
        self.log("listo - elapsed {} segundos en KNN".format(end - time_log))
        self.log("tiempo total: {}".format(end - beg))

        acc = accuracy_score(self.Y_test, y_pred)
        time = (end - beg) / 60.0

        score = self._score(time, acc)
        if self.logger:
            print("logging {} {} {} {}".format(state, acc, time, score))
            self.logger.log(state, acc, time, score)
            self.log("Evaluando: {} => Accuracy: {}, Time: {} minutos, Score: {}".format(state, acc, time, score))
            self.memoize_state[state] = score
        return score

    def _score(self, time, acc):
        """agregada para poder sobreescribirla en una clase hija de ser necesario"""
        if time < self.min_time:
            return acc / self.min_time  # divido por min_time para desempatar los valores del intervalo
        elif time < self.max_time:
            return acc / time
        return acc / (self.time_penalization * time) # acá tengo problemas con empates, pero fue

    def log(self, msg):
        if self.print_log:
            print(msg)

class KNNDecorator(KNNHyperParameters):

    def __init__(self, decorated):
        self.decorated = decorated

    def value(self, state):
        return self.decorated.value(state)

    def result(self, state, action):
        return self.decorated.result(state, action)

    def actions(self, state):
        return self.decorated.actions(state)

class KNNRandomStateDecorator(KNNDecorator):

    def __init__(self, decorated):
        super().__init__(decorated)

    def generate_random_state(self):
        """this method receives nothing, and must return a randomly generated state.

        Devuelvo un valor entre las dos dimesiones de X_train dividido
        por 10. Muestras de entrenamiento / 10 => max vecinos,
        vocabulario / 10 => max alfa"""

        return (random.randrange(1, int(self.decorated.X_train.shape[0] / 10)),
                random.randrange(1, int(self.decorated.X_train.shape[1] / 10)))


class KNNGridDecorator(KNNDecorator):
    """Genera estados iniciales a partir de una grilla y los devuelve en orden aleatorio
    Este decorator, toma una instancia del problema y le agrega el método generate_random_state

    Es para usar con algoritmos que requieren varios estados iniciales aleatorios, como beam.

    la escala de division está relacionada con el step y el beam size

    beam_size: es la cantidad de estados iniciales, divition_scale es
    el tamaño en cada dimensión. De esta manera si beam_size = 4
    divition_scale = 10, debería generar cuatro puntos distantes en
    10, en cada dimensión. Es divition_scale, porque escala por el
    aspect_ratio del step en cada dimensión de la búsqueda local. Esto
    es para que la busqueda local tenga la misma proba de cubrir el
    tamaño de cada grilla en cada dimensión
    """

    def __init__(self, decorated, seeders, divition_scale):
        print("Generando Grilla")
        super().__init__(decorated)
        aspect_ratio = decorated.pca_step / decorated.neightbours_step
        divitions = int(math.sqrt(seeders))+1
        self.grid = []
        for k in range(1, divitions * divition_scale, divition_scale):
            for alpha in range(2, divitions * divition_scale, int(divition_scale * aspect_ratio)):
                self.grid.append((k, alpha))
                self.metadata = [] # tendrá los valores de la grilla en el orden de popeo

    def generate_random_state(self):
        print("Devolviendo algún valor de la grilla")
        i = random.randrange(0, len(self.grid))
        point = self.grid.pop(i)
        self.metadata.append(point)
        return point

def sort_dataset_like_classify(df, df_test, _):
    """
    Casi choripasteado de classify, la única diferencia es que no uso ids_test; el label_test lo saco del evaluate.py
    Si bien recive un DF de entrenamiento, siempre usa imbd_small
    """
    text_train = df[df.type == 'train']["review"]
    label_train = df[df.type == 'train']["label"]

    text_test = df_test["review"]

    df_true = pd.read_csv("../data/test_sample.true")
    label_test = df_true["label"] # es el true
    #ids_test = df_test["id"]

    return text_train, label_train, text_test, label_test

def sort_dataset(df, df_test, data_set_cut):
    text_train = df[df.type == 'train']["review"]
    label_train = df[df.type == 'train']["label"]
    text_test = df_test[df.type == 'test']["review"]
    label_test = df_test[df.type == 'test']["label"]

    if args.data_set_cut:
        print("Achicando dataset al {}".format(data_set_cut))
        text_train = text_train[:int(len(text_train)*data_set_cut)]
        print(len(text_train))
        label_train = label_train[:int(len(label_train)*data_set_cut)]
        text_test = text_test[:int(len(text_test)*data_set_cut)]
        label_test = label_test[:int(len(label_test)*data_set_cut)]

    print("Class balance : {} pos {} neg".format(
        (label_train == 'pos').sum() / label_train.shape[0],
        (label_train == 'neg').sum() / label_train.shape[0]))

    return text_train, label_train, text_test, label_test

create_vectorizer_dispatcher = {}
def create_vectorizer_like_classify(text_train):
    vectorizer = CountVectorizer(
        max_df=0.85, min_df=0.01,
        max_features=5000, ngram_range=(1, 2),
    )

    vectorizer.fit(text_train)
    return vectorizer
create_vectorizer_dispatcher["like-classify"] = create_vectorizer_like_classify

def create_vectorizer_5000(text_train):
    vectorizer = CountVectorizer(max_df=0.85, min_df=0.01, max_features=5000)
    vectorizer.fit(text_train)
    return vectorizer
create_vectorizer_dispatcher["5000"] = create_vectorizer_5000

def create_vectorizer_sqrt(text_train):
    import re
    import math

    d={}
    reg=',|\n|;|\.| '
    for idx in text_train.keys():
        for word in re.split(reg, str(text_train[idx])):
            d[word]=1

    vocabulary = int(math.sqrt(sum(d.values())))*3

    vectorizer = CountVectorizer(max_df=0.85, min_df=0.01, max_features=min(vocabulary,5000))
    vectorizer.fit(text_train)

    return vectorizer
create_vectorizer_dispatcher["sqrt-tokens"] = create_vectorizer_sqrt

def file_name_suffix(args):
    file_suffix = str(args.algorithm)
    file_suffix += '_' + str(args.implementation)
    file_suffix += '_k:' + str(args.k)
    file_suffix += '_a:' + str(args.alpha)
    file_suffix += '_k-step:' + str(args.k_step)
    file_suffix += '_a-step:' + str(args.alpha_step)
    if str(args.implementation) in ["beam", "grid-beam"]:
        file_suffix += "_beam-size:" + str(args.beam_size)
    if args.like_classify:
        file_suffix += '_' + "like-classify.csv"
    else:
        file_suffix += '_vecto:' + args.vectorizer
        file_suffix += '_train:' + str(args.data_set_train.parts[-1])
        file_suffix += '_test:' + str(args.data_set_test.parts[-1])

    return file_suffix

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Hacer alguna busqueda local sobre los hiperparámetros de KNN.')
    parser.add_argument('-imp', '--implementation', choices=["sentiment", "sklearn"]
                        ,help='usar "sentiment" nuestra implementación de KNN y PCA o la de la biblioteca "sklearn"')
    parser.add_argument('-k', type=int, default=initial_neightbours_default
                        ,help='La cantidad inicial de vecinos a considerar - por defecto usa el de la clase')
    parser.add_argument('--alpha', type=int, default=None
                        ,help='La cantidad de componentes principales incial a considerar - por defecto usa el de la clase')
    parser.add_argument('--print-log', type=bool, default=print_log_default
                        ,help='Si imprime los logs a medida de que avanza - por defecto usa el de la clase')
    parser.add_argument('--k-step', type=int, default=neightbours_step_default
                        ,help='El tamaño del paso al moverse por el vecindario en la dimensión de vecinos - por defecto usa el de la clase')
    parser.add_argument('--alpha-step', type=int, default=pca_step_default
                        ,help='El tamaño del paso al moverse por el vecindario en la dimensión de componentes principales')
    parser.add_argument('--use-pca', dest='use_pca', action='store_true'
                        ,help='Indica que se usará PCA')
    parser.add_argument('--not-use-pca', dest='use_pca', action='store_false'
                        ,help='Indica que NO se usará PCA')
    parser.add_argument('--data-set-train', type=Path, default=dataset_train_default
                        ,help='Path del dataset de entrenamiento, puede ser relativo descomprimido - por defecto usa ../../data/imdb_small.csv')
    parser.add_argument('--data-set-test', type=Path, default=dataset_test_default
                        ,help='Path del dataset para hacer predicciones')
    parser.add_argument('--algorithm', choices=["hill-climbing", "beam", "grid-beam"], default="hill_climbing"
                        ,help='El algoritmo a usar para la búsqueda')
    parser.add_argument('--beam-size', type=int, default=beam_size_default
                        ,help='Si se usa beamer, la cantidad de estados iniciales que se considera - por defecto 10')
    parser.add_argument('--iterations_limit', type=int, default=None
                        ,help="Si se pasa, acota la cantidad de iteraciones - por defecto sigue hasta que no puede mejorar")
    parser.add_argument('--use-sparse-override', dest='use_sparse_override', action='store_true'
                        ,help='Le pasa matrices ralas a las funciones de knn y pca siempre')
    parser.add_argument('--use-dense-override', dest='use_sparse_override', action='store_false'
                        ,help='Le pasa matrices densas a las funciones de knn y pca siempre')
    parser.add_argument('--memoize-pca', dest='memoize_pca', action='store_true'
                        ,help='Indica que la busqueda local debe memoizar PCA.')
    parser.add_argument('--no-memoize-pca', dest='memoize_pca', action='store_true'
                        ,help='Indica si la busqueda local NO debe memoizar PCA.')
    parser.add_argument('--data-set-cut', type=float, default=None
                        ,help='Porcentaje del data set a utilizar')
    parser.add_argument('--divition-scale', type=int, default=divition_scale_default
                        ,help="La escala en cada dimensión de la grilla de estados iniciales")
    parser.add_argument('--out-history', type=Path, default=None
                        ,help='Path al archivo de salida donde se guardará la historia de la búsqueda')
    parser.add_argument('--out-metadata', type=Path, default=None
                        ,help='Path al archivo de salida donde se guardará metadata asociada al algoritmo, por ejemplo si se usa grid-beam, la grilla')
    parser.add_argument('--like-classify', dest='like_classify', action='store_true'
                        ,help='usa imbd_small y test_sample.true con los mismos parámetros que el classify')
    parser.add_argument('--vectorizer', choices=["like-classify", "5000", "sqrt-tokens"], default="5000")
    parser.add_argument('--ep', default=pca_eps_default
                        ,help='Tolerancia para el power method')
    parser.add_argument('--grid-k', nargs='+', default=None
                        ,help='Junto a grid-alpha crea el producto cartesiano con los valores pasados como lista')
    parser.add_argument('--grid-alpha', nargs='+', default=None
                        ,help='Junto a grid-a crea el producto cartesiano con los valores pasados como lista')
    parser.set_defaults(use_pca=usar_pca_default,use_sparse_override=None,memoize_pca=True,like_classify=False)

    args = parser.parse_args()

    # armo la grilla
    if args.grid_alpha is None:
        args.grid_alpha = [ args.alpha ]
    if args.grid_k is None:
        args.grid_k = [ args.grid_k ]

    the_grid = []
    for k in args.grid_k:
        for a in args.grid_alpha:
            the_grid.append((int(k), int(a)))

    # BEGIN CHORIPASTEO
    import pandas as pd

    #!cd ../../data && tar -xvf *.tgz
    #!cd ../../data && tar -xvf *.tar.gz

    df = pd.read_csv(args.data_set_train, index_col=0)
    df_test = pd.read_csv(args.data_set_test, index_col=0)

    print("Cantidad de documentos totales en el dataset: {}".format(df.shape[0]))

    from sklearn.feature_extraction.text import CountVectorizer
    if args.like_classify:
        print("Corriendo como classify")
        if not args.data_set_test == dataset_test_default:
            raise Exception("El dataset de test tiene que ser test_sample.csv, el default")
        text_train, label_train, text_test, label_test = sort_dataset_like_classify(df, df_test, args.data_set_cut)
        vectorizer = create_vectorizer_dispatcher["like-classify"](text_train)
    else:
        if args.data_set_test == dataset_test_default:
            print("No se aclaró set de test, usando el mismo de training y separando por etiquetas")
            df_test = df
            text_train, label_train, text_test, label_test = sort_dataset(df, df_test, args.data_set_cut)
            vectorizer = create_vectorizer_dispatcher[args.vectorizer](text_train)

    print("Cantidad de instancias de entrenamiento = {}".format(len(text_train)))
    print("Cantidad de instancias de test = {}".format(len(text_test)))

    # ENDCHORIPASTEO

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

    # diccionario global para que otras celdas aprovechen lo calculado
    if args.memoize_pca:
        print(the_grid)
        alphas = []
        for (k, a) in the_grid:
            if a not in alphas:
                alphas.append(a)
        alphas.sort()
        max_alpha = alphas[-1]
        if len(alphas) > 1:
            print("alphas: {}".format(alphas))
            max_alpha += max_alpha - alphas[-2] + args.alpha_step # el ancho de la grilla más el paso
        print("Precalculando un PCA de {} componentes con {} epsilon".format(max_alpha, args.ep))
        pca = pca_klass_constructor(max_alpha, args.ep)(max_alpha)
        pca.fit(X_train)
        x_train = pca.transform(X_train)
        x_test = pca.transform(X_test)
        pca_memoize = {}
        pca_memoize[max_alpha] = (x_train, x_test)
    else:
        pca_memoize = None

    # este genera una instancia de problema por grilla
    for (k, alpha) in the_grid:
        print("grid: ({}, {})".format(k, alpha))
        file_suffix = file_name_suffix(args)
        if not args.out_history:
            args.out_history="history_" + file_suffix
        if not args.out_metadata:
            args.out_metadata="metadata_" + file_suffix

        hpo_logger=None
        if args.out_history:
            hpo_logger = HPOLogger(args.out_history, ['k', 'alpha'])

        print("Creando Problema")
        knn_problem = KNNHyperParameters(X_train, y_train, X_test, y_test
                                         ,classifier_from=args.implementation, pca_from=args.implementation
                                         ,neightbours_step=args.k_step, pca_step=args.alpha_step
                                         ,initial_neightbours=k, initial_pca=alpha, pca_eps=args.ep
                                         ,usar_pca=args.use_pca, memoize_pca=pca_memoize,
                                         print_log=args.print_log, logger=hpo_logger)

        from simpleai.search.viewers import BaseViewer
        visor = BaseViewer()

        if args.algorithm == "hill-climbing":
            print("Resolviendo con Hill Climbing")
            from simpleai.search.local import hill_climbing
            result = hill_climbing(knn_problem, viewer=visor, iterations_limit=args.iterations_limit)
            print("Encontramos: {}\nLuego de este camino: {}\n".format(result.state, result.path()))

        elif args.algorithm == "beam":
            print("Resolviendo con Beam")
            from simpleai.search.local import beam
            knn_problem = KNNRandomStateDecorator(knn_problem)
            result = beam(knn_problem, viewer=visor, beam_size=args.beam_size, iterations_limit=args.iterations_limit)
            print("Encontramos: {}\nLuego de este camino: {}\n".format(result.state, result.path()))

        elif args.algorithm == "grid-beam":
            print("Resolviendo con Beam no random, engrillado")
            from simpleai.search.local import beam
            knn_problem = KNNGridDecorator(knn_problem, args.beam_size, args.divition_scale)
            result = beam(knn_problem, viewer=visor, beam_size=args.beam_size, iterations_limit=args.iterations_limit)
            print("Encontramos: {}\nLuego de este camino: {}\n".format(result.state, result.path()))

        if knn_problem.metadata:
            with open(args.out_metadata, 'w') as metadata_file:
                if args.algorithm == "grid-beam":
                    metadata_file.write("K, PCA")
                    for point in knn_problem.metadata:
                        k, alfa = point
                        metadata_file.write(str(k) + ", " + str(alfa) + "\n")

