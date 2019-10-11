#pragma once

#include <Eigen/Sparse>
#include <Eigen/Dense>

using Eigen::MatrixXd;

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix;
typedef Eigen::SparseMatrix<double> SparseMatrix;

typedef Eigen::VectorXd Vector;
typedef Eigen::SparseVector<double> SparseVector;

typedef enum e_Criterion
{
    all = 0,
    eigenvalues = 1,
    residual_vector = 2,
    eigenvectors = 3
} Criterion;
