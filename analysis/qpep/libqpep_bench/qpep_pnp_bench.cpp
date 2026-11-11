// qpep_pnp_bench.cpp
//
// 用 LibQPEP 的真实 C++ 实现测量 QPEP 的两种模式：
//   full   : solver_WQ          (596x596 模板, 40 个候选, 论文 (17) 的完整 Gröbner)
//   approx : solver_WQ_approx   (249x249 模板, 27 个候选, 论文 (24) 的 lambda=0 约化系统)
// 对 approx 还跑 4 组 row-permutation 分支（论文说的 3-of-4），以及 LibQPEP demo 里
// 实际使用的 3 行求和变体。
//
// 输出 CSV 到 stdout。每个 trial 一行 × 每种方法。
//
// 这是本报告作者写的 benchmark driver；LibQPEP 本身未作算法修改（仅裁剪了不相关的
// covariance/CSDP/OpenCV 目标以便在无 LAPACK/OpenCV 的环境里编译）。

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "pnp_WQD.h"
#include "QPEP_grobner.h"
#include "QPEP_lm_single.h"
#include "solver_WQ.h"
#include "solver_WQ_approx.h"
#include "misc_pnp_funcs.h"
#include "generateProjectedPoints.h"
#include "utils.h"

using Eigen::Matrix3d;
using Eigen::Matrix4d;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Vector4d;

static double rotErrDeg(const Matrix3d& R, const Matrix3d& R0) {
    double c = (R.transpose() * R0).trace();
    c = std::max(-1.0, std::min(1.0, (c - 1.0) / 2.0));
    return std::acos(c) * 180.0 / M_PI;
}

struct Out {
    // 每次求解的输出
    Matrix3d R;
    Vector3d t;
    double loss = 1e15;
    double tDec = 0, tGro = 0, tEig = 0, tLM = 0;
    int status = -1;
};

int main(int argc, char** argv) {
    int n_trials = 50;
    int npts = 25;
    double noise_px = 1.0;
    int seed0 = 0;
    int with_lm = 1;
    std::string data_file;
    double problem_scale = 1e-8;
    int skip_full = 0;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--trials")) n_trials = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--points")) npts = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--noise")) noise_px = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--seed")) seed0 = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--no-lm")) with_lm = 0;
        else if (!std::strcmp(argv[i], "--data")) data_file = argv[++i];
        else if (!std::strcmp(argv[i], "--scale")) problem_scale = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--skip-full")) skip_full = 1;
    }

    Matrix3d K0 = [] {
        Matrix3d k;
        k << 810.27, 0, 568.66,
             0, 825.58, 321.98,
             0, 0, 1;
        return k;
    }();
    // 可选：使用 LibQPEP 自带 demo 的几何（世界点/相机内参/位姿），更接近作者配置
    Matrix3d Rfile;
    Vector3d tfile;
    Matrix3d Kfile;
    std::vector<Vector3d> world_file;
    std::vector<Vector2d> image_file;
    bool use_file = !data_file.empty();
    if (use_file) {
        readPnPdata(data_file, Rfile, tfile, Kfile, world_file, image_file);
        std::fprintf(stderr, "[dbg] read %s: %zu world pts, K=(%.1f,%.1f,%.1f,%.1f)\n",
                     data_file.c_str(), world_file.size(), Kfile(0, 0), Kfile(1, 1), Kfile(0, 2), Kfile(1, 2));
    }

    std::printf("trial,method,npts,noise,rot_deg,trans_err,loss,t_dec,t_gro,t_eig,t_lm,total,status\n");
    int npts_eff = npts;

    for (int tr = 0; tr < n_trials; ++tr) {
        std::mt19937_64 gen(seed0 * 1000003ULL + tr);
        std::uniform_real_distribution<double> uni(-1.0, 1.0);
        std::normal_distribution<double> gauss(0.0, 1.0);

        Matrix3d K = use_file ? Kfile : K0;
        // 世界点：默认立方体内随机；--data 时用 demo 几何
        std::vector<Vector3d> world_pt;
        if (use_file) {
            world_pt = world_file;
            npts_eff = (int)world_pt.size();
        } else {
            world_pt.resize(npts);
            for (int i = 0; i < npts; ++i)
                world_pt[i] = Vector3d(uni(gen), uni(gen), uni(gen));
            npts_eff = npts;
        }

        Matrix3d R0;
        Vector3d t0;
        if (use_file) {
            R0 = Rfile;
            t0 = tfile;
        } else {
            Vector3d axis(gauss(gen), gauss(gen), gauss(gen));
            axis.normalize();
            double angle = M_PI * (0.2 + 0.6 * std::fabs(uni(gen)));
            R0 = Eigen::AngleAxisd(angle, axis).toRotationMatrix();
            t0 = Vector3d(1.0 + 1.5 * uni(gen), 0.5 * uni(gen), 3.0 + 1.0 * uni(gen));
        }

        std::vector<Vector2d> image_pt;
        std::vector<double> depth;
        if (use_file)
            image_pt = image_file;  // demo 文件里的像点与 (R0,t0,world,K) 自洽
        else
            generateProjectedPoints(image_pt, depth, world_pt, K, R0, t0);

        // 像素噪声
        for (auto& p : image_pt) {
            p.x() += noise_px * gauss(gen);
            p.y() += noise_px * gauss(gen);
        }

        // QPEP 系数矩阵
        Eigen::Matrix<double, 4, 64> W;
        Eigen::Matrix<double, 4, 4> Q;
        Eigen::Matrix<double, 3, 37> D;
        Eigen::Matrix<double, 4, 24> coef_f_q_sym;
        Eigen::Matrix<double, 1, 70> coef_J_pure;
        Eigen::Matrix<double, 3, 10> coefs_tq;
        Eigen::Matrix<double, 3, 3> pinvG;
        W.setZero(); Q.setZero(); D.setZero(); coef_f_q_sym.setZero();
        coef_J_pure.setZero(); coefs_tq.setZero(); pinvG.setZero();
        pnp_WQD(W, Q, D, coef_f_q_sym, coef_J_pure, coefs_tq, pinvG,
                image_pt, world_pt, K, problem_scale);

        if (tr == 0) {
            std::fprintf(stderr, "[dbg] npts=%zu nimg=%zu W00=%g W01=%g Q00=%g\n",
                         world_pt.size(), image_pt.size(), W(0, 0), W(0, 1), Q(0, 0));
            std::fprintf(stderr, "[dbg] |W|=%.3g |Q|=%.3g |coefJ|=%.3g pinvG=%.3g %.3g %.3g\n",
                         W.norm(), Q.norm(), coef_J_pure.norm(), pinvG(0,0), pinvG(1,1), pinvG(2,2));
            std::fprintf(stderr, "[dbg] img0 = %.3f %.3f  world0 = %.3f %.3f %.3f\n",
                         image_pt[0].x(), image_pt[0].y(), world_pt[0].x(), world_pt[0].y(), world_pt[0].z());
        }
        auto loss_of = [&](const Matrix3d& R, const Vector3d& t) {
            Vector4d q = R2q(R);
            Eigen::Quaterniond qq(q(3), q(0), q(1), q(2));
            std::vector<double> mon = mon_J_pure_pnp_func(qq, t);
            double J = 0;
            for (int i = 0; i < 70; ++i) J += coef_J_pure(0, i) * mon[i];
            return J;
        };

        auto run = [&](const char* method, const std::string& module,
                       const Eigen::MatrixXd& Wm, const Eigen::MatrixXd& Qm,
                       const int* perm) -> Out {
            Out o;
            QPEP_options opt;
            opt.ModuleName = module;
            opt.DecompositionMethod = "PartialPivLU";
            Eigen::Matrix4d X;
            double minv[40];
            QPEP_runtime stat = [&] {
                if (module == "solver_WQ")
                    return QPEP_WQ_grobner(o.R, o.t, X, minv, Wm, Qm,
                                           reinterpret_cast<solver_func_handle>(solver_WQ),
                                           reinterpret_cast<mon_J_pure_func_handle>(mon_J_pure_pnp_func),
                                           reinterpret_cast<t_func_handle>(t_pnp_func),
                                           coef_J_pure, coefs_tq, pinvG, perm, opt);
                else
                    return QPEP_WQ_grobner(o.R, o.t, X, minv, Wm, Qm,
                                           reinterpret_cast<solver_func_handle>(solver_WQ_approx),
                                           reinterpret_cast<mon_J_pure_func_handle>(mon_J_pure_pnp_func),
                                           reinterpret_cast<t_func_handle>(t_pnp_func),
                                           coef_J_pure, coefs_tq, pinvG, perm, opt);
            }();
            o.status = stat.statusGrobner == 0 ? 0 : -1;
            o.tDec = stat.timeDecomposition;
            o.tGro = stat.timeGrobner;
            o.tEig = stat.timeEigen;
            o.loss = loss_of(o.R, o.t);
            if (with_lm && o.status == 0) {
                Vector4d q0 = R2q(o.R);
                stat = QPEP_lm_single(o.R, o.t, X, q0, 100, 5e-2,
                                      reinterpret_cast<eq_Jacob_func_handle>(eq_Jacob_pnp_func),
                                      reinterpret_cast<t_func_handle>(t_pnp_func),
                                      coef_f_q_sym, coefs_tq, pinvG, stat);
                o.tLM = stat.timeLM;
                o.loss = loss_of(o.R, o.t);
            }
            return o;
        };

        auto emit = [&](const char* name, const Out& o) {
            std::fprintf(stderr, "[dbg] emit %s\n", name);
            std::printf("%d,%s,%d,%.4g,%.6g,%.6g,%.9g,%.6g,%.6g,%.6g,%.6g,%.6g,%d\n",
                        tr, name, npts_eff, noise_px,
                        rotErrDeg(o.R, R0), (o.t - t0).norm(), o.loss,
                        o.tDec, o.tGro, o.tEig, o.tLM,
                        o.tDec + o.tGro + o.tEig + o.tLM, o.status);
        };

        // 1) full
        std::fprintf(stderr, "[dbg] trial %d loss_gt=%.4g\n", tr, loss_of(R0, t0));
        {
            Out o = run("full", "solver_WQ", W, Q, nullptr);
            emit("full", o);
        }

        // 2) approx：LibQPEP demo 实际使用的 3 行求和变体（W_.row0=W0+W1+W2, ...）
        std::fprintf(stderr, "[dbg] start approx_sum\n");
        {
            Eigen::Matrix<double, 3, 64> W3;
            Eigen::Matrix<double, 3, 4> Q3;
            W3.row(0) = W.row(0) + W.row(1) + W.row(2);
            W3.row(1) = W.row(1) + W.row(2) + W.row(3);
            W3.row(2) = W.row(2) + W.row(3) + W.row(0);
            Q3.row(0) = Q.row(0) + Q.row(1) + Q.row(2);
            Q3.row(1) = Q.row(1) + Q.row(2) + Q.row(3);
            Q3.row(2) = Q.row(2) + Q.row(3) + Q.row(0);
            Out o = run("approx_sum", "solver_WQ_approx", W3, Q3, nullptr);
            emit("approx_sum", o);
        }

        // 3) approx 的四组 3-of-4（论文的四个 row permutations）
        std::fprintf(stderr, "[dbg] start perms\n");
        {
            const int perms[4][3] = {{0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3}};
            const char* names[4] = {"approx_p012", "approx_p013", "approx_p023", "approx_p123"};
            for (int k = 0; k < 4; ++k) {
                Out o = run(names[k], "solver_WQ_approx", W, Q, perms[k]);
                emit(names[k], o);
            }
        }
    }
    return 0;
}
