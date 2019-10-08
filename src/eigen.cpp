#include <algorithm>
#include <chrono>
#include <iostream>
#include "eigen.h"

using namespace std;


pair<double, Vector> power_iteration(const Matrix& A, unsigned num_iter, double eps)
{
    Vector v = Vector::Random(A.cols());
    double lambda = 0;
    Vector Av = A*v;

    double prev_lambda = 0;
    Vector prev_v = Vector(A.cols());
    Vector residual = Vector(A.cols());
    for (unsigned i = 0; i < num_iter; ++i)
    {
        prev_v = v;
        v = Av/Av.norm();
        Av = A*v;

        prev_lambda = lambda;
        lambda = (v.transpose()*Av);
        lambda /= v.norm();

        if ((prev_v-v).norm() < eps) {
            std::cout << "[c1] corta PM por diferencia entre autovectores en iteración: " << i+1 << "/" << num_iter << '\n';
            break;
        }
        if (std::abs(prev_lambda-lambda) < eps) {
            std::cout << "[c2] corta PM por diferencia entre autovalores en iteración: " << i+1 << "/" << num_iter << '\n';
            break;
        }
        residual = Av - lambda*v;
        if ((residual).norm() < eps) {
            std::cout << "[c3] corta PM por diferencia residual en iteración: " << i+1 << "/" << num_iter << '\n';
            break;
        }
    }

    return make_pair(lambda, v);
}

pair<Vector, Matrix> get_first_eigenvalues(const Matrix& X, unsigned num, unsigned num_iter, double epsilon)
{
    Matrix A(X);
    Vector eigvalues(num);
    Matrix eigvectors(A.rows(), num);

    for(unsigned i = 0; i < num; ++i){
        std::cout << "i / num: " << i+1 << " / " << num << '\n';
        auto eigvalue_vector = power_iteration(A, num_iter, epsilon);
        eigvalues(i) = eigvalue_vector.first;
        eigvectors.col(i) = eigvalue_vector.second;
        A = A - eigvalue_vector.first * eigvalue_vector.second * eigvalue_vector.second.transpose();
    }

    return make_pair(eigvalues, eigvectors);
}
