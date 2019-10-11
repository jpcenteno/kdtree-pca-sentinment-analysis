'''
Este modulo provee un wrapper con KNN junto a PCA.
'''

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

class PCAKneighboursClasifier(BaseEstimator, ClassifierMixin):

    def __init__(self, k=100, alpha=100):
        self._k = k
        self._alpha = alpha

        self._pca = PCA(alpha)
        self._knn = KNeighborsClassifier(k)

    def fit(self, X, y):
        self._pca.fit(X)
        X = self._pca.transform(X)
        self._knn.fit(X, y)

    def predict(self, X, y=None):
        X = self._pca.transform(X)
        return self._knn.predict(X)
