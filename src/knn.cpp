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
    //#ifdef LLVL1
    cout << "test : " << classify.rows() << " " << classify.cols() << endl;
    cout << "model: " << _X.rows() << " " << _X.cols() << endl;
    //#endif
    _kd_tree.query(classify, _neighbors, idx, dists);

    //get the class that appears more often than the others within the first k elements
    vector<int> classesCounter(2,0);  //two classes (neg = 0, pos = 1)
    Vector maxClassVals(classify.cols());
    for (unsigned int i = 0; i < classify.cols(); i++)
        maxClassVals[i] = -1; // TODO ver de usar inicializador
    int maxCounter = -1;

    for(unsigned int j = 0; j < classify.cols(); j++){
        for(unsigned int i = 0; i < _neighbors; ++i){
            auto classVal = _y(idx(i, j), 0);
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
