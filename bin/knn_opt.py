import sys
import pandas as pd
import numpy as np

sys.path.append("notebooks/")

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from classify import get_instances

if __name__ == '__main__':
    df = pd.read_csv("./data/imdb_small.csv")

    text_train = df[df.type == 'train']["review"]
    label_train = df[df.type == 'train']["label"]

    text_test = df[df.type == 'test']["review"]
    label_test = df[df.type == 'test']["label"]


    print("Vectorizando datos...")
    vectorizer = CountVectorizer(max_df=0.90, min_df=0.01, max_features=5000)

    vectorizer.fit(text_train)

    X_train, y_train = vectorizer.transform(text_train), (label_train == 'pos').values
    X_test, y_test = vectorizer.transform(text_test), (label_test == 'pos').values

    vals = []
    for k in range(1, 3000, 30):
        print("rep: {}".format(k))

        """
        Entrenamos KNN
        """

        skl_clf = KNeighborsClassifier(k).fit(X_train, y_train)
        """
        Testeamos
        """
        print("Prediciendo etiquetas...")
        y_pred = skl_clf.predict(X_test)

        acc = accuracy_score(y_pred, y_test)
        vals.append([k, acc])

    df = pd.DataFrame(vals, columns=["vecinos", "precision"])
    df.to_csv("data/k_opt.csv", index=False)

    sns.set(font_scale=1.5)
    ax = sns.lineplot(x="vecinos", y="precision", err_style="band", data=df)
    plt.show()
    fig = ax.get_figure()
    fig.set_size_inches(18,6)

    fig.savefig("knn.png")
    fig.clear()


    print(df)
