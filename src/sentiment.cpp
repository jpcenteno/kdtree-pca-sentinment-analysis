#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "knn.h"
#include "pca.h"
#include "eigen.h"

namespace py=pybind11;

// el primer argumento es el nombre...
PYBIND11_MODULE(sentiment, m) {
    py::enum_<Criterion>(m, "Criterion", py::arithmetic())
        .value("all", all)
        .value("eigenvalues", eigenvalues)
        .value("residual_vector", residual_vector)
        .value("eigenvectors", eigenvectors)
        .export_values();

    py::class_<KNNClassifier>(m, "KNNClassifier")
        .def(py::init<unsigned int>())
        .def("fit", &KNNClassifier::fit)
        .def("predict", &KNNClassifier::predict);

    py::class_<PCA>(m, "PCA")
        .def(py::init<unsigned int, Criterion>())
        .def(py::init<unsigned int>())
        .def("fit", &PCA::fit)
        .def("transform", &PCA::transform)
        .def("fit_transform", &PCA::fit_transform);
    m.def(
        "power_iteration", &power_iteration,
        "Function that calculates eigenvector",
        py::arg("X"),
        py::arg("criterion")="all",
        py::arg("num_iter")=5000,
        py::arg("epsilon")=0.001
    );
    m.def(
        "get_first_eigenvalues", &get_first_eigenvalues,
        "Function that calculates eigenvector",
        py::arg("X"),
        py::arg("num"),
        py::arg("criterion")="all",
        py::arg("num_iter")=5000,
        py::arg("epsilon")=0.001
    );

}
