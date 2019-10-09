#pragma once

#include "kdtree_eigen.h"
#include "types.h"

using namespace kdt;

class KNNClassifier {
public:
    KNNClassifier(unsigned int n_neighbors);

    void fit(SparseMatrix X, Matrix y);

    Vector predict(SparseMatrix X);
private:
    double predictVector(const Vector &) const;

    KDTree<double>::Matrix _X;
    KDTree<double> _kd_tree;
    Matrix _y;
    unsigned int _neighbors;

};
