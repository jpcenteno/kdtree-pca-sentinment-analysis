"""

python3 bin/classify_pm_eps.py data/test_sample.csv test_sample
python3 bin/evaluate.py data/test_sample_<resto>.out data/test_sample.true

"""
# Estas dos líneas permiten que python encuentre la librería sentiment en notebooks/
import sys
sys.path.append("notebooks/")

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sentiment import PCA, KNNClassifier, Criterion
from classify import get_instances
import time

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Uso: python classify_pm_eps.py archivo_de_test archivo_salida")
        exit()

    test_path = sys.argv[1]
    out_name = sys.argv[2]

    df = pd.read_csv("data/imdb_small.csv")
    df_test = pd.read_csv(test_path)

    print("Vectorizando datos...")
    X_train_orig, y_train, X_test_orig, ids_test = get_instances(df, df_test)
    alpha = 100
    X_train_orig = X_train_orig.toarray()

    exponentes = {
        6: Criterion.eigenvalues,
        7: Criterion.eigenvalues
        #9: Criterion.eigenvalues,
        #11: Criterion.all
    }
    paths = {}
    for exp in exponentes:
        paths[exp] = []
        print("exponente: {}".format(exp))
        for rep in range(1):
            print("rep: {}".format(rep))

            out_path = "data/{}_{}_{}.out".format(out_name, exp, rep)
            paths[exp].append(out_path)

            eps = 10**(-exp)
            pca = PCA(alpha, eps)

            print("Entrenando PCA")
            t = time.clock()
            pca.fit(X_train_orig)

            print("Transformando datos")
            X_train = pca.transform(X_train_orig)
            X_test = pca.transform(X_test_orig)

            total_time = time.clock() - t
            print("time: {}".format(total_time))

            """
            Entrenamos KNN
            """

            clf = KNNClassifier(5)
            print("Entrenando KNN")
            clf.fit(X_train, y_train)

            """
            Testeamos
            """
            print("Prediciendo etiquetas...")
            y_pred = clf.predict(X_test).reshape(-1)
            # Convierto a 'pos' o 'neg'
            labels = ['pos' if val == 1 else 'neg' for val in y_pred]

            df_out = pd.DataFrame({"id": ids_test, "label": labels})
            df_out.to_csv(out_path, index=False)

            print("Salida guardada en {}".format(out_path))
    print(paths)
