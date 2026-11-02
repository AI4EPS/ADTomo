#include <torch/extension.h>
#include <vector>
#include <algorithm>
#include <tuple>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <set>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>

// The 2D counterpart of Eikonal3D.cpp, line for line: m, n count nodes; the source is at fractional grid coordinates
// (x, y); the 4 corners of its cell start from Simpson's rule and stay fixed; sweeps stop when no node changes by more
// than tol (s) or after MAX_ITER iterations.

typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;

#define u(i, j) u[(i) * n + (j)]
#define f(i, j) f[(i) * n + (j)]
#define get_id(i, j) ((i) * n + (j))

const int MAX_ITER = 100;

double calculate_unique_solution(double a1_, double a2_, double f, double h)
{
    double a1 = a1_, a2 = a2_;
    if (a1 > a2)
        std::swap(a1, a2);

    double x = a1 + f * h;
    if (x <= a2)
        return x;
    double B = -(a1 + a2);
    double C = (a1 * a1 + a2 * a2 - f * f * h * h) / 2.0;
    x = (-B + sqrt(B * B - 4 * C)) / 2.0;
    return x;
}

void sweeping_over_I_J(double *u, const double *f, int m, int n, double h, int ix0, int jx0, int ix1, int jx1, int dirI, int dirJ)
{

    auto I = std::make_tuple(dirI == 1 ? 0 : m - 1, dirI == 1 ? m : -1, dirI);
    auto J = std::make_tuple(dirJ == 1 ? 0 : n - 1, dirJ == 1 ? n : -1, dirJ);

    for (int i = std::get<0>(I); i != std::get<1>(I); i += std::get<2>(I))
        for (int j = std::get<0>(J); j != std::get<1>(J); j += std::get<2>(J))
        {
            if (i == ix0 && j == jx0) continue;
            if (i == ix0 && j == jx1) continue;
            if (i == ix1 && j == jx0) continue;
            if (i == ix1 && j == jx1) continue;

            double uxmin = i == 0 ? u(i + 1, j) : (i == m - 1 ? u(i - 1, j) : std::min(u(i + 1, j), u(i - 1, j)));
            double uymin = j == 0 ? u(i, j + 1) : (j == n - 1 ? u(i, j - 1) : std::min(u(i, j + 1), u(i, j - 1)));
            double u_new = calculate_unique_solution(uxmin, uymin, f(i, j), h);
            u(i, j) = std::min(u_new, u(i, j));
        }
}

void sweeping(double *u, const double *f, int m, int n, double h, int ix0, int jx0, int ix1, int jx1)
{
    sweeping_over_I_J(u, f, m, n, h, ix0, jx0, ix1, jx1, 1, 1);
    sweeping_over_I_J(u, f, m, n, h, ix0, jx0, ix1, jx1, -1, 1);
    sweeping_over_I_J(u, f, m, n, h, ix0, jx0, ix1, jx1, -1, -1);
    sweeping_over_I_J(u, f, m, n, h, ix0, jx0, ix1, jx1, 1, -1);
}

void forward(double *u, const double *f, double h,
             int m, int n, double x, double y, double tol = 1e-8)
{
    int ix0 = std::max(0, std::min((int)floor(x), m - 2));  // m, n count nodes: the last cell starts at m - 2
    int jx0 = std::max(0, std::min((int)floor(y), n - 2));
    int ix1 = ix0 + 1;
    int jx1 = jx0 + 1;
    for (int i = 0; i < m * n; i++)
    {
        u[i] = 100000.0;
    }

    // Simpson's Rule t = (d/6) * (f_src + 4*f_mid + f_tgt)
    double f00 = f(ix0, jx0);
    double f01 = f(ix0, jx1);
    double f10 = f(ix1, jx0);
    double f11 = f(ix1, jx1);

    // Bilinear interpolation weights at source
    double wx = x - ix0;
    double wy = y - jx0;
    double fsrc = (1-wx)*(1-wy)*f00 + (1-wx)*wy*f01
                + wx*(1-wy)*f10 + wx*wy*f11;

    // Slowness at the center of the grid cell
    double fmid = (f00 + f01 + f10 + f11) / 4.0;

    // Traveltimes to each corner using Simpson's Rule
    double d00 = sqrt((x - ix0) * (x - ix0) + (y - jx0) * (y - jx0)) * h;
    u(ix0, jx0) = (d00 / 6.0) * (fsrc + 4.0 * fmid + f00);

    double d01 = sqrt((x - ix0) * (x - ix0) + (y - jx1) * (y - jx1)) * h;
    u(ix0, jx1) = (d01 / 6.0) * (fsrc + 4.0 * fmid + f01);

    double d10 = sqrt((x - ix1) * (x - ix1) + (y - jx0) * (y - jx0)) * h;
    u(ix1, jx0) = (d10 / 6.0) * (fsrc + 4.0 * fmid + f10);

    double d11 = sqrt((x - ix1) * (x - ix1) + (y - jx1) * (y - jx1)) * h;
    u(ix1, jx1) = (d11 / 6.0) * (fsrc + 4.0 * fmid + f11);

    auto u_old = new double[m * n];
    bool converged = false;
    for (int i = 0; i < MAX_ITER; i++)
    {
        memcpy(u_old, u, sizeof(double) * m * n);
        sweeping(u, f, m, n, h, ix0, jx0, ix1, jx1);
        double err = 0.0;

        for (int j = 0; j < m * n; j++)
        {
            err = std::max(fabs(u[j] - u_old[j]), err);
        }

        if (err < tol)
        {
            converged = true;
            break;
        }
    }
    delete[] u_old;
    if (!converged)
        printf("Warning: Eikonal2D did not converge in %d iterations\n", MAX_ITER);
}

void backward(
    double *grad_f,
    const double *grad_u,
    const double *u, const double *f, double h,
    int m, int n, double x, double y)
{

    int ix0 = std::max(0, std::min((int)floor(x), m - 2));
    int jx0 = std::max(0, std::min((int)floor(y), n - 2));
    int ix1 = ix0 + 1;
    int jx1 = jx0 + 1;

    Eigen::VectorXd g(m * n);
    memcpy(g.data(), grad_u, sizeof(double) * m * n);

    // calculate gradients for \partial L/\partial f
    Eigen::VectorXd rhs(m * n);
    // should not include the boundary, overwrite below
    for (int i = 0; i < m * n; i++)
    {
        rhs[i] = -2 * f[i] * h * h;
    }

    std::vector<T> triplets;
    std::set<int> zero_id;
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {

            int this_id = get_id(i, j);

            if ((i == ix0 && j == jx0) || (i == ix0 && j == jx1) || (i == ix1 && j == jx0) || (i == ix1 && j == jx1))
            {
                triplets.push_back(T(this_id, this_id, 1.0));
                continue;
            }

            double uxmin = i == 0 ? u(i + 1, j) : (i == m - 1 ? u(i - 1, j) : std::min(u(i + 1, j), u(i - 1, j)));
            double uymin = j == 0 ? u(i, j + 1) : (j == n - 1 ? u(i, j - 1) : std::min(u(i, j + 1), u(i, j - 1)));

            int idx = i == 0 ? get_id(i + 1, j) : (i == m - 1 ? get_id(i - 1, j) : (u(i + 1, j) > u(i - 1, j) ? get_id(i - 1, j) : get_id(i + 1, j)));
            int idy = j == 0 ? get_id(i, j + 1) : (j == n - 1 ? get_id(i, j - 1) : (u(i, j + 1) > u(i, j - 1) ? get_id(i, j - 1) : get_id(i, j + 1)));

            bool this_id_is_not_zero = false;
            if (u(i, j) > uxmin)
            {
                this_id_is_not_zero = true;
                triplets.push_back(T(this_id, this_id, 2.0 * (u(i, j) - uxmin)));
                triplets.push_back(T(this_id, idx, -2.0 * (u(i, j) - uxmin)));
            }

            if (u(i, j) > uymin)
            {
                this_id_is_not_zero = true;
                triplets.push_back(T(this_id, this_id, 2.0 * (u(i, j) - uymin)));
                triplets.push_back(T(this_id, idy, -2.0 * (u(i, j) - uymin)));
            }

            if (!this_id_is_not_zero)
            {
                zero_id.insert(this_id);
                g[this_id] = 0.0;
            }
        }
    }

    // Adjoint of the discrete equations G(u) = f^2 h^2 (derivation in Eikonal3D.cpp): solve G'(u)^T x = grad_u,
    // then grad_f = -x * dG/df.
    if (zero_id.size() > 0)
    {
        printf("Warning: zero_id.size() = %d\n", (int)zero_id.size());
        for (auto &t : triplets)
        {
            if (zero_id.count(t.col()) || zero_id.count(t.row()))
                t = T(t.col(), t.row(), 0.0);
        }
        for (auto idx : zero_id)
        {
            triplets.push_back(T(idx, idx, 1.0));
        }
    }

    SpMat A(m * n, m * n);
    A.setFromTriplets(triplets.begin(), triplets.end());
    A = A.transpose();
    Eigen::SparseLU<SpMat> solver;

    solver.analyzePattern(A);
    solver.factorize(A);
    Eigen::VectorXd res = solver.solve(g);
    for (int i = 0; i < m * n; i++)
    {
        grad_f[i] = -res[i] * rhs[i];
    }

    // Simpson's Rule gradient
    // u_ij = (d_ij/6) * (fsrc + 4*fmid + ftgt), fmid the mean of the 4 corners
    // ∂u_ij/∂f_kl = (d_ij/6) * (w_kl + 1 + δ_{ij,kl}) where w_kl is the bilinear weight
    double wx = x - ix0;
    double wy = y - jx0;
    double w00 = (1-wx)*(1-wy);
    double w01 = (1-wx)*wy;
    double w10 = wx*(1-wy);
    double w11 = wx*wy;

    double res00 = res[get_id(ix0, jx0)];
    double res01 = res[get_id(ix0, jx1)];
    double res10 = res[get_id(ix1, jx0)];
    double res11 = res[get_id(ix1, jx1)];

    double d00 = sqrt((x - ix0) * (x - ix0) + (y - jx0) * (y - jx0)) * h;
    double d01 = sqrt((x - ix0) * (x - ix0) + (y - jx1) * (y - jx1)) * h;
    double d10 = sqrt((x - ix1) * (x - ix1) + (y - jx0) * (y - jx0)) * h;
    double d11 = sqrt((x - ix1) * (x - ix1) + (y - jx1) * (y - jx1)) * h;

    // grad_f[kl] = Σ_ij res_ij * (d_ij/6) * (w_kl + 1 + δ_{ij,kl})
    grad_f[get_id(ix0, jx0)] = (res00 * d00 * (w00 + 1 + 1)
                              + res01 * d01 * (w00 + 1)
                              + res10 * d10 * (w00 + 1)
                              + res11 * d11 * (w00 + 1)) / 6.0;

    grad_f[get_id(ix0, jx1)] = (res00 * d00 * (w01 + 1)
                              + res01 * d01 * (w01 + 1 + 1)
                              + res10 * d10 * (w01 + 1)
                              + res11 * d11 * (w01 + 1)) / 6.0;

    grad_f[get_id(ix1, jx0)] = (res00 * d00 * (w10 + 1)
                              + res01 * d01 * (w10 + 1)
                              + res10 * d10 * (w10 + 1 + 1)
                              + res11 * d11 * (w10 + 1)) / 6.0;

    grad_f[get_id(ix1, jx1)] = (res00 * d00 * (w11 + 1)
                              + res01 * d01 * (w11 + 1)
                              + res10 * d10 * (w11 + 1)
                              + res11 * d11 * (w11 + 1 + 1)) / 6.0;

}

// PyTorch extension interface
torch::Tensor eikonal_forward(torch::Tensor f, double h, double x, double y)
{
    TORCH_CHECK(f.dim() == 2, "f must be a 2D tensor");
    TORCH_CHECK(f.is_contiguous(), "Input tensors must be contiguous");

    int m = f.size(0);
    int n = f.size(1);

    auto u = torch::zeros_like(f);

    forward(u.data_ptr<double>(), f.data_ptr<double>(), h, m, n, x, y);

    return u;
}

torch::Tensor eikonal_backward(torch::Tensor grad_u, torch::Tensor u, torch::Tensor f, double h, double x, double y)
{
    TORCH_CHECK(grad_u.dim() == 2 && u.dim() == 2 && f.dim() == 2, "All tensors must be 2D");
    TORCH_CHECK(grad_u.sizes() == u.sizes(), "All tensors must have the same size");
    TORCH_CHECK(grad_u.is_contiguous() && u.is_contiguous() && f.is_contiguous(), "All tensors must be contiguous");

    int m = u.size(0);
    int n = u.size(1);

    auto grad_f = torch::zeros_like(f);

    backward(grad_f.data_ptr<double>(),
             grad_u.data_ptr<double>(), u.data_ptr<double>(), f.data_ptr<double>(),
             h, m, n, x, y);

    return grad_f;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &eikonal_forward, "Eikonal2D forward");
    m.def("backward", &eikonal_backward, "Eikonal2D backward");
}
