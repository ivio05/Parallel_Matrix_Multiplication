#include <omp.h>
#include <vector>

template <typename T>
void parallel_matrix_multiplication(int p, int M, int N, int K,\
const std::vector< std::vector<T> >(& a), const std::vector< std::vector<T> >(& b), std::vector< std::vector<T> > (& c));

template <typename T>
void consistent_matrix_multiplication(int M, int N, int K,\
const std::vector< std::vector<T> >(& a), const std::vector< std::vector<T> >(& b), std::vector< std::vector<T> > (& c));