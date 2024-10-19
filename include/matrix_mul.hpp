#pragma once

#include <omp.h>
#include <vector>

template <typename T>
void parallel_matrix_multiplication(int p, int M, int N, int K,\
const std::vector< std::vector<T> >(& a), const std::vector< std::vector<T> >(& b), std::vector< std::vector<T> > (& c)) {
    #pragma omp parallel for num_threads(p) collapse(2) schedule(guided, K)
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            T el = 0;
            for (int n = 0; n < N; n++) {
                el += a[m][n] * b[n][k];
            }
            c[m][k] = el;
        }
    }
}

template <typename T>
void consistent_matrix_multiplication(int M, int N, int K,\
const std::vector< std::vector<T> >(& a), const std::vector< std::vector<T> >(& b), std::vector< std::vector<T> > (& c)) {
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            T el = 0;
            for (int n = 0; n < N; n++) {
                el += a[m][n] * b[n][k];
            }
            c[m][k] = el;
        }
    }
}