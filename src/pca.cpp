#include <iostream>
#include "pca.h"
#include "eigen.h"

using namespace std;


PCA::PCA(unsigned int n_components) : _nComponents(n_components)
{

}

void PCA::fit(SparseMatrix X)
{
    _covMatrix = getCovariance(X);
    _eigenvalues_vectors = get_first_eigenvalues(_covMatrix, _nComponents);
}

MatrixXd PCA::transform(SparseMatrix X)
{
    return X * _eigenvalues_vectors.second;
}

MatrixXd PCA::fit_transform(SparseMatrix X)
{
    fit(X);
    return transform(X);
}

Matrix PCA::getMedias(const SparseMatrix &M) const {
    unsigned int n = M.rows();
    unsigned int m = M.row(0).cols();
    Matrix medias(1,m);

    for(unsigned int i = 0; i < n; ++i){
        medias.row(0) = medias.row(0) + M.row(i);
    }

    return medias/n;
}

///@param M: Matrix nxm
///@param medias : 1xm
///return M_0 - medias
///       ...
///       M_n - medias
Matrix PCA::get_M_Minus_Medias(const SparseMatrix &M, const Matrix &medias) const {
    unsigned int n = M.rows();
    unsigned int m = medias.cols();
    Matrix M_minus_medias(n, m);

    for(unsigned int i = 0; i < n; ++i){
        M_minus_medias.row(i) = M.row(i)-medias.row(0);
    }

    return M_minus_medias;
}


Matrix PCA::getCovariance(const SparseMatrix &M)const {
    Matrix medias = getMedias(M);
    Matrix M_minus_medias = get_M_Minus_Medias(M, medias);
    return (M_minus_medias.transpose()*M_minus_medias)/(M.rows()-1);
}
