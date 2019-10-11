#include <algorithm>
//#include <chrono>
#include "knn.h"
#include <iostream>
#include <queue>

using namespace std;
using namespace kdt;

KNNClassifier::KNNClassifier(unsigned int n_neighbors) : _neighbors(n_neighbors)
{
}

void KNNClassifier::setNeightbors(unsigned int neightbors) {
    _neighbors = neightbors;
}

void KNNClassifier::fit(SparseMatrix X, Matrix y)
{
    _X = X.toDense().transpose(); // cada columna es un dato
    _kd_tree.setData(_X);
    _kd_tree.build();
    _y = y;
}

Vector KNNClassifier::predict(SparseMatrix X)
{
    KDTreed::Matrix classify = X.toDense().transpose();
    KDTreed::Matrix dists; // basically Eigen::MatrixXd
    KDTreed::MatrixI idx;  // basically Eigen::Matrix<Eigen::Index>
    #ifdef LLVL1
    cout << "test : " << classify.rows() << " " << classify.cols() << endl;
    cout << "model: " << _X.rows() << " " << _X.cols() << endl;
    cout << "tree:  " << _kd_tree.dimension() << " " << _kd_tree.size() << endl;
    cout << "y:     " << _y.rows() << " " << _y.cols() << endl;
    #endif
    _kd_tree.query(classify, _neighbors, idx, dists);

    Vector maxClassVals(idx.cols()); // tantas filas como puntos
    // idx KNNxM
    for (unsigned int j = 0; j < idx.cols(); ++j) {
        int maxCounter = -1;
        maxClassVals[j] = -1;
        vector<int> classesCounter(2, 0); // two classes (neg = 0, pos = 1)
        for (unsigned int i = 0; i < idx.rows(); ++i) {
            auto classVal = _y(idx(i, j), 0);
#ifdef LLVL2
            cout << "indice de " << i << "-vecino: " << classVal << " label: " << classVal << endl;
#endif
            classesCounter[classVal]++;
            if (classesCounter[classVal] > maxCounter) {
                maxCounter = classesCounter[classVal];
                maxClassVals[j] = classVal;
            }
        }
    }

    //return the class
    return maxClassVals;
}
