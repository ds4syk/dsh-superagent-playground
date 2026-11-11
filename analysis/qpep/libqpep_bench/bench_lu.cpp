// bench_lu.cpp: 量 LibQPEP 两种消除模板的纯线性代数代价
// full:   596x596 LU + 40 RHS + 40x40 eigen
// approx: 249x249 LU + 27 RHS + 27x27 eigen
#include <cstdio>
#include <chrono>
#include <vector>
#include <algorithm>
#include <random>
#include <Eigen/Core>
#include <Eigen/Dense>

using clk = std::chrono::high_resolution_clock;

template <int n, int k>
double bench(int reps) {
    std::mt19937_64 gen(42);
    std::normal_distribution<double> g(0, 1);
    Eigen::MatrixXd A(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) A(i, j) = g(gen);
    Eigen::MatrixXd B(n, k);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < k; ++j) B(i, j) = g(gen);
    A = A + n * Eigen::MatrixXd::Identity(n, n);
    std::vector<double> ts;
    for (int r = 0; r < reps; ++r) {
        auto t0 = clk::now();
        Eigen::PartialPivLU<Eigen::MatrixXd> lu(A);
        Eigen::MatrixXd X = lu.solve(B);
        Eigen::MatrixXd C = A.topRows(k) * A.leftCols(k);
        Eigen::EigenSolver<Eigen::MatrixXd> es(C);
        volatile double s = X.sum() + es.eigenvalues().real().sum();
        (void)s;
        auto t1 = clk::now();
        ts.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(ts.begin(), ts.end());
    return ts[ts.size() / 2];
}

int main() {
    int reps = 30;
    double full = bench<596, 40>(reps);
    double appr = bench<249, 27>(reps);
    std::printf("full_template_ms,%.3f\n", full);
    std::printf("approx_template_ms,%.3f\n", appr);
    std::printf("ratio,%.2f\n", full / appr);
    return 0;
}
