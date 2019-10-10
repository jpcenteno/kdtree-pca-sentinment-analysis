"""
grafica la evolucion de precision y tiempo del metodo de la potencia

python eps_opt.py iteraciones_por_rango [opcional: repeticiones_por_epsilon max_exponente_epsilon]
"""

import sys
sys.path.append("notebooks/")

import random
import pandas as pd
import numpy as np
import signal
import re
import time
import datetime


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sentiment import Criterion, power_iteration
import seaborn as sns
import matplotlib.pyplot as plt
from classify import get_instances

crits_to_name = {
    Criterion.residual_vector: "vector residual",
    Criterion.eigenvectors: "autovectores",
    Criterion.eigenvalues: "autovalores"
}

def signal_handler(signal, frame):
    global global_benchmarks, crit_benchmarks

    print('You pressed Ctrl+C - or killed me with -2')
    global_benchmarks += crit_benchmarks

    df = pd.DataFrame(global_benchmarks, columns = ["criterio", "exponente", "precision", "tiempo (ticks)"])
    print(df)
    df.to_csv("data/eps_opt_out.csv", index=False)

    with open("data/eps_opt_out.csv", 'a') as file:
        file.write(",\n argumentos: {}".format(sys.argv))
        file.write(",")
    sys.exit(0)

def get_acc(X, crit, eps):
    l, v = power_iteration(A, crit, num_iter=15000, epsilon=eps)
    acc = 1/np.linalg.norm(A@v - l*v)
    return acc

def rate(A, crit, eps):
    t = time.clock()
    acc = get_acc(A, crit, eps)
    total_t = time.clock()-t
    return acc, total_t

def avg(l):
    return sum(l)/len(l)

def run_iters(A, iters, crit, eps_iters, max_exp):
    global crit_benchmarks

    print("Criterio: {}".format(crit))
    crit_benchmarks = []
    for i in range(iters+1):
        print("iteracion {}".format(i))
        # de 1e0 a 1e-max_exp
        exp = i*(max_exp)/iters
        eps = 10**(-exp)
        acc_list = []
        time_list = []
        for j in range(eps_iters):
            acc, time = rate(A, crit, eps)
            crit_benchmarks.append([crits_to_name[crit], exp, acc, time])
    return crit_benchmarks

if __name__ == '__main__':
    global global_benchmarks, crit_benchmarks

    crit_benchmarks = []
    global_benchmarks = []
    signal.signal(signal.SIGINT, signal_handler)

    if len(sys.argv) < 2:
        print("Uso: python eps_opt.py iteraciones_por_rango [opcional: repeticiones_por_epsilon  max_exponente_epsilon  sample_size]")
        exit()

    rango_iters = int(sys.argv[1])
    eps_iters = int(sys.argv[2]) if len(sys.argv) > 2 is not None else 3
    max_exp = int(sys.argv[3]) if len(sys.argv) > 2 is not None else 8

    if True:
        df = pd.read_csv("./data/imdb_small.csv")

        print("Vectorizando datos...")

        text_train = df[df.type == 'train']["review"]
        label_train = df[df.type == 'train']["label"]

        vectorizer = CountVectorizer(
            max_df=0.75, min_df=0.1,
            max_features=5000, ngram_range=(1, 2),
        )

        vectorizer.fit(text_train)

        X, y = vectorizer.transform(text_train), (label_train == 'pos').values

        #recortamos la muestra
        X = X.toarray()

        sample_size = int(sys.argv[4]) if len(sys.argv) > 4 is not None else X.shape[0]
        if sample_size > X.shape[0]:
            print("sample_size muy grande, tiene que ser <= a {}".format(X.shape[0]))
        print(sample_size)
        indexes = random.sample([i for i in range(X.shape[0])], sample_size)
        X = [X[i,:] for i in indexes]

        #numpy nos da la matriz de covarianza
        A = np.cov(X)

        print("Midiendo parametros...")
        for crit in crits_to_name:
            global_benchmarks += run_iters(A, rango_iters, crit, eps_iters, max_exp)

        df = pd.DataFrame(global_benchmarks, columns = ["criterio", "exponente", "precision", "tiempo (ticks)"])
    else:
        df = pd.read_csv("corrida/eps_opt_out.csv")
        df = df.loc[(df['exponente'] < 12)]

    def plot(y_col):
        ax = sns.lineplot(x="exponente", y=y_col, hue="criterio", err_style="band", data=df)
        plt.show()
        fig = ax.get_figure()
        fig.set_size_inches(18,6)

        ts = int(time.time())
        d = str(datetime.datetime.fromtimestamp(ts))

        fig.savefig(y_col.split()[0] + "_{}".format("".join(re.split("-| |:", d))) + ".png")
        fig.clear()


    df.to_csv("data/eps_opt_out.csv", index=False)
    print(df)

    sns.set(font_scale=1.5)
    plot("precision")
    plot("tiempo (ticks)")
