# coding=utf-8
"""
Tests de eigen, knn y pca
correr como
python3 test.py
"""
# Estas dos líneas permiten que python encuentre la librería sentiment en notebooks/
import sys
sys.path.append("../../notebooks/")

import pandas as pd
import numpy as np
import math
import traceback
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.datasets import fetch_openml

# PCA, KNNClassifier, power_iteration, get_first_eigenvalues
import sentiment as st

#copiado de los tests del taller1
def mntest(func):
    global tests

    tests.append(func)

    return func

def correr_tests():
    excepciones = []
    for test in tests:
        try:
            print("Corriendo {} ... ".format(test.__name__), end='')
            test()
            print("OK")
        except AssertionError as e:
            error_msg = traceback.format_exc()
            excepciones.append((test, error_msg))
            print("ERROR")

    if len(excepciones) > 0:
        print("\nErrores:\n")
        for (test, error_msg) in excepciones:
            print("En {}".format(test.__name__))
            print(error_msg)
    else:
        print("\n\nTodos los tests pasaron correctamente")

tests = []


"""
Power Method
"""

@mntest
def testear_pm_diagonal_3x3():

    A = np.diag([1.,2.,3.])
    A_reversed = np.diag([1.,2.,3.][::-1])

    c1, b1 =  st.power_iteration(A, st.Criterion.all, num_iter=15000, epsilon=1e-8)
    c2, b2 =  st.power_iteration(A_reversed, st.Criterion.all, num_iter=15000, epsilon=1e-8)

    # norma 1
    assert(abs(np.linalg.norm(b1)-1) < 1e-4)
    assert(abs(np.linalg.norm(b2)-1) < 1e-4)
    # es autovector y autovalor
    assert(np.allclose(A@b1, c1*b1, rtol = 1e-4 , atol = 1e-6))
    assert(np.allclose(A_reversed@b2, c2*b2, rtol = 1e-4 , atol = 1e-6))
    # es el autovalor más grande
    assert(abs(c1 - 3.) < 1e-15)
    assert(abs(c2 - 3.) < 1e-15)

@mntest
def testear_pm_diagonal_4x4():
    A = np.diag([1.,2.,3.,4.])
    A_reversed = np.diag([1.,2.,3.,4.][::-1])

    c1, b1 =  st.power_iteration(A, st.Criterion.all, num_iter=15000, epsilon=1e-8)
    c2, b2 =  st.power_iteration(A_reversed, st.Criterion.all, num_iter=15000, epsilon=1e-8)

    # norma 1
    assert(abs(np.linalg.norm(b1)-1) < 1e-4)
    assert(abs(np.linalg.norm(b2)-1) < 1e-4)
    # es autovector y autovalor
    assert(np.allclose(A@b1, c1*b1, rtol = 1e-4 , atol = 1e-6))
    assert(np.allclose(A_reversed@b2, c2*b2, rtol = 1e-4 , atol = 1e-6))
    # es el autovalor más grande
    assert(abs(c1 - 4.) < 1e-15)
    assert(abs(c2 - 4.) < 1e-15)

@mntest
def testear_pm_diagonalizable_5x5():
    d = np.diag([1 ,2 ,3 ,4 ,5])
    v = np.array([
        [1,0,0,0,0],
        [2,2,3,0,0],
        [3,0,2,3,1],
        [4,4,2,0,0],
        [5,4,2,0,1]
    ])

    A = v @ d @ v.T
    eigenvals, _ = np.linalg.eig(A)
    c1, b1 =  st.power_iteration(A, st.Criterion.all, num_iter=15000, epsilon=1e-8)

    # norma 1
    assert(abs(np.linalg.norm(b1)-1) < 1e-15)
    # es autovector y autovalor
    assert(np.allclose(A@b1, c1*b1))
    # es el autovalor más grande
    assert(abs(c1 - eigenvals[0]) < 1e-13)


"""
Deflación
"""

@mntest
def testear_deflacion_diagonalizable_5x5():
    d = np.diag([1 ,2 ,3 ,4 ,5])
    v = np.array([
        [1,0,0,0,0],
        [2,2,3,0,0],
        [3,0,2,3,1],
        [4,4,2,0,0],
        [5,4,2,0,1]
    ])

    A = v @ d @ v.T
    eigenvals, _ = np.linalg.eig(A)
    c, b =  st.get_first_eigenvalues(A, 5, st.Criterion.all, num_iter=15000)

    # norma 1
    for i in range(5):
        assert(abs(np.linalg.norm(b[:,i])-1) < 1e-15)
    # es autovector y autovalor
    for i in range(5):
        assert(np.allclose(A@b[:,i], c[i]*b[:,i]))
    # es el autovalor más grande
    assert(np.allclose(sorted(eigenvals, reverse=True), c))

@mntest
def testear_deflacion_diagonalizable_6x6():
    d = np.diag([1 ,2 ,3 ,4 ,5, 6])
    v = np.array([
        [1,0,0,0,0,0],
        [2,2,3,0,0,0],
        [3,0,2,3,1,0],
        [4,4,2,0,0,0],
        [3,1,2,0,0,1],
        [5,4,2,0,1,0]
    ])
    A = v @ d @ v.T
    eigenvals, _ = np.linalg.eig(A)
    c, b =  st.get_first_eigenvalues(A, 6, st.Criterion.all, num_iter=15000)

    # norma 1
    for i in range(6):
        assert(abs(np.linalg.norm(b[:,i])-1) < 1e-15)
    # es el autovalor más grande
    assert(np.allclose(sorted(eigenvals, reverse=True), c))
    # es autovector y autovalor
    for i in range(6):
        assert(np.allclose(A@b[:,i], c[i]*b[:,i]))
"""
KNN PCA
"""
@mntest
def testear_pca_knn():
    # si da accuracy 100 al ser testeada con su training set entonces hace lo que se pide al entrenarse y testear

    X = [
        [1, 0],
        [0, 1],
        [3, 0],
        [5, 2],
        [2, 1],
        [2, 10],
        [20, 10],
        [25, 10],
    ]
    y = [i%2 for i in range(len(X))]

    alpha = 5
    pca = st.PCA(alpha)

    pca.fit(X, 0.0001)
    X_trans = pca.transform(X)

    clf = st.KNNClassifier(1)
    X = X_trans
    clf.fit(X, y)

    y_pred = clf.predict(X).reshape(-1)

    acc = accuracy_score(y, y_pred)

    print("Accuracy: {}".format(acc))
    assert(acc > 0.99999)
    return

if __name__ == '__main__':
    correr_tests()
    #mismo_test()
