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

void KNNClassifier::fit(SparseMatrix X, Matrix y)
{
    _X = X.toDense().transpose(); // cada columna es un dato
    _kd_tree.setData(_X);
    _kd_tree.build();
    _y = y;
}


Vector KNNClassifier::predict(SparseMatrix X)
{
    // Creamos vector columna a devolver
    auto ret = Vector(X.rows());

    for (unsigned k = 0; k < X.rows(); ++k)
    {
        ret(k) = predictVector(X.row(k));
    }
    #ifdef LLVL1
    cout << "predijimos todos!" << endl;
    #endif
    return ret;
}

double KNNClassifier::predictVector(const Vector &testVector) const{

    KDTreed::Matrix dists; // basically Eigen::MatrixXd
    KDTreed::MatrixI idx;  // basically Eigen::Matrix<Eigen::Index>
    #ifdef LLVL1
    cout << "test : " << testVector.rows() << " " << testVector.cols() << endl;
    cout << "model: " << _X.rows() << " " << _X.cols() << endl;
    #endif
    _kd_tree.query(testVector, _neighbors, idx, dists);

    //get the class that appears more often than the others within the first k elements
    vector<int> classesCounter(2,0);  //two classes (neg = 0, pos = 1)
    int maxClassVal = -1;
    int maxCounter = -1;

    for(unsigned int i = 0; i < _neighbors; ++i){
        auto classVal = _y(idx(i), 0);
        classesCounter[classVal]++;
        if(classesCounter[classVal] > maxCounter){
            maxCounter = classesCounter[classVal];
            maxClassVal = classVal;
        }
    }

    //return the class
    return maxClassVal;
}
