#pragma once
#include "types.h"

class PCA {
public:
    PCA(unsigned int n_components);

    // TODO: pasar a SparseMatrix - Este Seguro que es el problema
    void fit(SparseMatrix X);

    Eigen::MatrixXd transform(SparseMatrix X);

    Eigen::MatrixXd fit_transform(SparseMatrix X);
private:
    Matrix getMedias(const SparseMatrix&) const;
    Matrix get_M_Minus_Medias(const SparseMatrix&, const Matrix&) const;
    Matrix getCovariance(const SparseMatrix &)const;

    unsigned int _nComponents;
    std::pair<Vector, Matrix> _eigenvalues_vectors;
    Matrix _covMatrix;

};
