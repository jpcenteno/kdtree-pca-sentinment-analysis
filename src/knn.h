#pragma once

#include "types.h"


class KNNClassifier {
public:
    KNNClassifier(unsigned int n_neighbors);

    void fit(SparseMatrix X, Matrix y);

    Vector predict(SparseMatrix X);
private:
    double predictVector(const Vector &) const;
    
    SparseMatrix _X;
    Matrix _y;
    unsigned int _neighbors;

};
