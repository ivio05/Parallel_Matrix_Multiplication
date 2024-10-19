#include "matrix_mul.hpp"

#include <omp.h>
#include <vector>
#include <iostream>
#include <random>
#include <typeinfo>
#include <chrono>
#include <sstream>
#include <string.h>
#include <fstream>

#define MAX_THREADS omp_get_max_threads()
#define FILENAME "results/results.csv"

template<typename T>
void generate_vector (int m, int n, std::vector<std::vector<T> > (& v), int min_value = -100, int max_value = 100) {
    std::uniform_real_distribution<T> unif(min_value, max_value);
    std::default_random_engine re;

    for (int i = 0; i < m; ++i) {
        std::vector<T> row;
        for (int j = 0; j < n; ++j) {
            row.push_back(unif(re));
        }
        v.push_back(row);
    }
}

void generate_vector (int m, int n, std::vector<std::vector<int> > (& v), int min_value = -100, int max_value = 100) {
    std::uniform_int_distribution unif(min_value, max_value);
    std::default_random_engine re;

    for (int i = 0; i < m; ++i) {
        std::vector<int> row;
        for (int j = 0; j < n; ++j) {
            row.push_back(unif(re));
        }
        v.push_back(row);
    }
}

template <typename T>
double test_parallel(int p, int m, int n, int k, std::vector<std::vector<T> > (& a), std::vector<std::vector<T> > (& b))
{
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    std::vector<std::vector<T> > c(m, std::vector<T> (k, 0));

    auto t1 = high_resolution_clock::now();
    parallel_matrix_multiplication<T>(p, m, n, k, a, b, c);
    auto t2 = high_resolution_clock::now();

    duration<double, std::milli> ms = t2 - t1;

    return double(ms.count());
}

template <typename T>
double test_consistent(int m, int n, int k, std::vector<std::vector<T> > (& a), std::vector<std::vector<T> > (& b))
{
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    std::vector<std::vector<T> > c(m, std::vector<T> (k, 0));

    auto t1 = high_resolution_clock::now();
    consistent_matrix_multiplication<T>(m, n, k, a, b, c);
    auto t2 = high_resolution_clock::now();

    duration<double, std::milli> ms = t2 - t1;

    return double(ms.count());
}

template <typename T>
std::vector<double> test_all_threads(int m, int n, int k, int min_value = -100, int max_value = 100) {
    std::vector<std::vector<T> > a;
    std::vector<std::vector<T> > b;
    generate_vector(m, n, a, min_value, max_value);
    generate_vector(n, k, b, min_value, max_value);
    std::cout << "Generated\n";

    std::vector<double> res;
    for (int p = 1; p <= MAX_THREADS; ++p) {
        double test_time = test_parallel(p, m, n, k, a, b);
        std::cout << "\nOn parametrs: m = " << m << ", n = " << n << ", k = " << k << ", p = " << p << ", type = " << typeid(T).name() << std::endl; 
        std::cout << "Parallel time: " << test_time << "ms\n";
        res.push_back(test_time);
    }

    double test_time = test_consistent(m, n, k, a, b);
    std::cout << "\nOn parametrs: m = " << m << ", n = " << n << ", k = " << k << ", type = " << typeid(T).name() << std::endl; 
    std::cout << "Consistent time: " << test_time << "ms\n";
    res.push_back(test_time);

    return res;
}

template <typename T>
std::vector<double> test_all_threads_average(int m, int n, int k) {
    std::vector<double> res1 = test_all_threads<T>(m, n, k);
    std::vector<double> res2 = test_all_threads<T>(m, n, k);
    std::vector<double> res3 = test_all_threads<T>(m, n, k);

    std::vector<double> results;
    for (int i = 0; i < MAX_THREADS + 1; ++i) {
        results.push_back((res1[i] + res2[i] + res3[i]) / 3);
    }

    return results;
}

void write_results_in_file(std::vector<double> results) {
    std::ofstream myfile;
    myfile.open("FILENAME", std::ios::app);
    for (int i = 0; i < MAX_THREADS + 1; ++i) {
        myfile << results[i] << ";";
    }
    myfile << "\n";
    myfile.close();
}

template <typename T>
void full_test(void) {
    std::vector<double> results;

    results = test_all_threads_average<T>(256, 256, 256);
    write_results_in_file(results);

    results = test_all_threads_average<T>(512, 512, 512);
    write_results_in_file(results);

    results = test_all_threads_average<T>(512, 1024, 512);
    write_results_in_file(results);

    results = test_all_threads_average<T>(1024, 1024, 1024);
    write_results_in_file(results);

    results = test_all_threads_average<T>(1024, 2048, 1024);
    write_results_in_file(results);

    results = test_all_threads_average<T>(2048, 2048, 2048);
    write_results_in_file(results);
}

int main(int argc, char const *argv[])
{

    if (argc == 2) {
        if ((strcmp(argv[1], "int") == 0)) {
            full_test<int>();
        } else if ((strcmp(argv[1], "float") == 0)) {
            full_test<float>();
        } else if ((strcmp(argv[1], "double") == 0)) {
            full_test<double>();
        } else {
            std::cerr << "Wrong type, only support int, float and double\n";
            return 1;
        }
        return 0;
    } else if (argc < 2 || argc > 5) {
        std::cerr << "Wrong number of args (1 for full testing {type}, and 4 for all thread test {m, n, k, type})\n";
        return 1;
    }
    int m, n, k;

    std::stringstream ss1(argv[1]);
    ss1 >> m;

    std::stringstream ss2(argv[2]);
    ss2 >> n;

    std::stringstream ss3(argv[3]);
    ss3 >> k;

    if ((strcmp(argv[4], "int") == 0)) {
        test_all_threads<int>(m, n, k);
    } else if ((strcmp(argv[4], "float") == 0)) {
        test_all_threads<float>(m, n, k);
    } else if ((strcmp(argv[4], "double") == 0)) {
        test_all_threads<double>(m, n, k);
    } else {
        std::cerr << "Wrong type, only support int, float and double\n";
        return 1;
    }
    return 0;


}
