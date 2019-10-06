# coding=utf-8
"""
"""
# Estas dos líneas permiten que python encuentre la librería sentiment en notebooks/
import sys
sys.path.append("notebooks/")

import pandas as pd
import numpy as np
import math
import traceback

from sentiment import PCA, KNNClassifier, power_iteration, get_first_eigenvalues

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

    np.linalg.eig(A)

    #assert()



correr_tests()
