#include <algorithm>
//#include <chrono>
#include <iostream>
#include <queue>
#include "knn.h"

using namespace std;


KNNClassifier::KNNClassifier(unsigned int n_neighbors) : _neighbors(n_neighbors)
{
}

void KNNClassifier::fit(SparseMatrix X, Matrix y)
{
    _X = X;
    _y = y;
}


Vector KNNClassifier::predict(SparseMatrix X)
{
    // Creamos vector columna a devolver
    auto ret = Vector(X.rows());

    for (unsigned k = 0; k < X.rows(); ++k)
    {
        ret(k) = predictVector(X.row(k));
    }

    return ret;
}

double KNNClassifier::predictVector(const Vector &testVector) const{

    auto testTranspose = testVector.transpose();
    //get norms of each vector in X minus testVector
    vector<pair<double, int>> norms;
    norms.reserve(_X.rows());

    for(unsigned int i = 0; i < _X.rows(); ++i) {
        norms.push_back({ (_X.row(i) - testTranspose).norm(), i });
    }

    typedef pair<double, int> priorityQueueElements;
    auto comp = [](const priorityQueueElements &l, const priorityQueueElements &r){
        return l.first > r.first;
    };

    //build priority queue of norms (the top element is the smaller)
    priority_queue<priorityQueueElements, vector<priorityQueueElements>, decltype(comp)> pqNorms(norms.begin(), norms.end(), comp);

    //get the class that appears more often than the others within the first k elements
    vector<int> classesCounter(2,0);  //two classes (neg = 0, pos = 1)
    int maxClassVal = -1;
    int maxCounter = -1;

    for(unsigned int i = 0; i < _neighbors; ++i){
        auto element = pqNorms.top();
        auto classVal = _y(element.second, 0);
        classesCounter[classVal]++;
        if(classesCounter[classVal] > maxCounter){
            maxCounter = classesCounter[classVal];
            maxClassVal = classVal;
        }
        pqNorms.pop();
    }

    //return the class
    return maxClassVal;
}
