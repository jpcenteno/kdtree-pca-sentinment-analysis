#include <algorithm>
#include <chrono>
#include <iostream>
#include "eigen.h"

using namespace std;


pair<double, Vector> power_iteration(const Matrix& X, unsigned num_iter, double eps)
{
    Vector b = Vector::Random(X.cols());
    double eigenvalue;
    for (unsigned i = 0; i < num_iter; ++i)
    {
        // cout << X << endl;
        // cout << "X: " << X.norm() << endl;
        const Vector& tmp = X*b;
        cout << "tmp: " << tmp.norm() << endl;
        const Vector& next_b = tmp / tmp.norm();
        cout << "next_b: " << next_b.norm() << endl;
        double delta = (next_b - b).norm();
        cout << "delta: " << delta << endl;
        if (delta < eps) {
            std::cout << "corta PM por diferencia" << '\n';
            b = next_b;
            break;
        }
        b = next_b;
    }

    eigenvalue = (b.transpose()*X*b);
    eigenvalue /= b.squaredNorm();

    return make_pair(eigenvalue, b / b.norm());
}

pair<Vector, Matrix> get_first_eigenvalues(const Matrix& X, unsigned num, unsigned num_iter, double epsilon)
{
    Matrix A(X);
    Vector eigvalues(num);
    Matrix eigvectors(A.rows(), num);

    for(unsigned i = 0; i < num; ++i){
        std::cout << "i / num: " << i << " / " << num << '\n';
        auto eigvalue_vector = power_iteration(A, num_iter, epsilon);
        eigvalues(i) = eigvalue_vector.first;
        eigvectors.col(i) = eigvalue_vector.second;
        A = A - eigvalue_vector.first * eigvalue_vector.second * eigvalue_vector.second.transpose();
    }

    return make_pair(eigvalues, eigvectors);
}
