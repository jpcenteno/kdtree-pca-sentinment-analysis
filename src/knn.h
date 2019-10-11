#pragma once

#include "kdtree_eigen.h"
#include "types.h"

using namespace kdt;

class KNNClassifier {
public:
    KNNClassifier(unsigned int n_neighbors);

    void fit(SparseMatrix X, Matrix y);

    Vector predict(SparseMatrix X);

    void setNeightbors(unsigned int neightbors);

private:

    KDTreed::Matrix _X;
    KDTreed _kd_tree;
    Matrix _y;
    unsigned int _neighbors;

};
