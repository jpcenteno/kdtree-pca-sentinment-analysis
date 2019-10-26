#include <algorithm>
#include <chrono>
#include <iostream>
#include "eigen.h"
#include <math.h>

using namespace std;


pair<double, Vector> power_iteration(const Matrix& A,  Criterion crit, unsigned num_iter, double eps)
{
    //comparamos cuadrados de normas porque es más barato
    eps = pow(eps, 2.0);
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

        if (crit != eigenvectors) {
            prev_lambda = lambda;
            lambda = (v.transpose()*Av);
            lambda /= v.norm();

            if (crit != residual_vector && std::abs(prev_lambda-lambda) < eps) {
                #ifdef LLVL1
                std::cout << "[c2] corta PM por diferencia entre autovalores en iteración: "
                          << i+1 << "/" << num_iter << " con eps: " << eps << '\n';
                #endif
                break;
            }
            residual = Av - lambda*v;
            if (crit != eigenvalues && (residual).squaredNorm() < eps) {
                #ifdef LLVL1
                std::cout << "[c3] corta PM por diferencia residual en iteración: "
                          << i+1 << "/" << num_iter << " con eps: " << eps << '\n';
                #endif
                break;
            }
        }
        if ((crit == all || crit == eigenvectors) && (prev_v-v).squaredNorm() < eps) {
            #ifdef LLVL1
            std::cout << "[c1] corta PM por diferencia entre autovectores en iteración: "
                      << i+1 << "/" << num_iter << " con eps: " << eps << '\n';
            #endif
            break;
        }
    }
    if (crit == eigenvectors) {
        lambda = (v.transpose()*Av);
        lambda /= v.norm();
    }

    return make_pair(lambda, v);
}

pair<Vector, Matrix> get_first_eigenvalues(const Matrix& X, unsigned num, Criterion crit, unsigned num_iter, double epsilon)
{
    Matrix A(X);
    Vector eigvalues(num);
    Matrix eigvectors(A.rows(), num);

    for(unsigned i = 0; i < num; ++i){
        #ifdef LLVL1
        std::cout << "i / num: " << i+1 << " / " << num << '\n';
        #endif
        auto eigvalue_vector = power_iteration(A, crit, num_iter, epsilon);
        eigvalues(i) = eigvalue_vector.first;
        eigvectors.col(i) = eigvalue_vector.second;
        A = A - eigvalue_vector.first * eigvalue_vector.second * eigvalue_vector.second.transpose();
    }

    return make_pair(eigvalues, eigvectors);
}
