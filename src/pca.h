#pragma once
#include "types.h"

class PCA {
public:
    PCA(unsigned int n_components);

    void fit(Matrix X);

    Eigen::MatrixXd transform(SparseMatrix X);
private:
    Matrix getMedias(const Matrix&) const;
    Matrix get_M_Minus_Medias(const Matrix&, const Matrix&) const;
    Matrix getCovariance(const Matrix &)const;

    unsigned int _nComponents;
    std::pair<Vector, Matrix> _eigenvalues_vectors;
    Matrix _covMatrix;

};
