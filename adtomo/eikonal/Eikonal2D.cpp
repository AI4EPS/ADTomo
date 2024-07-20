#include <torch/extension.h>
#include <vector>
#include <algorithm>
#include <vector>
#include <iostream>
#include <utility>

// #include "../eigen/Eigen/Core"
// #include "../eigen/Eigen/SparseCore"
// #include "../eigen/Eigen/SparseLU"
#include "Eigen/Core"
#include "Eigen/SparseCore"
#include "Eigen/SparseLU"

typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;

double solution(double a, double b, double f, double h)
{
  double d = fabs(a - b);
  if (d >= f * h)
    return std::min(a, b) + f * h;
  else
    return (a + b + sqrt(2 * f * f * h * h - (a - b) * (a - b))) / 2;
}

void sweep(double *u,
           const std::vector<int> &I, const std::vector<int> &J,
           const double *f, int m, int n, double h)
{
  for (int i : I)
  {
    for (int j : J)
    {
      double a, b;
      if (i == 0)
      {
        a = u[j * (m + 1) + 1];
      }
      else if (i == m)
      {
        a = u[j * (m + 1) + m - 1];
      }
      else
      {
        a = std::min(u[j * (m + 1) + i + 1], u[j * (m + 1) + i - 1]);
      }
      if (j == 0)
      {
        b = u[(m + 1) + i];
      }
      else if (j == n)
      {
        b = u[(n - 1) * (m + 1) + i];
      }
      else
      {
        b = std::min(u[(j - 1) * (m + 1) + i], u[(j + 1) * (m + 1) + i]);
      }
      double u_new = solution(a, b, f[j * (m + 1) + i], h);
      u[j * (m + 1) + i] = std::min(u[j * (m + 1) + i], u_new);
    }
  }
}

// void forward(double *u, const double *f, int m, int n, double h, int ix, int jx)
void forward(double *u, const double *f, int m, int n, double h, double x, double y)
{
  int ix0 = std::max(0, std::min((int)floor(x / h), m - 1));
  int jx0 = std::max(0, std::min((int)floor(y / h), n - 1));
  int ix1 = ix0 + 1;
  int jx1 = jx0 + 1;
  for (int i = 0; i < m + 1; i++)
  {
    for (int j = 0; j < n + 1; j++)
    {
      u[j * (m + 1) + i] = 100000.0;
      // if (i == ix && j == jx)
      //   u[j * (m + 1) + i] = 0.0;
    }
  }
  // interpolate
  u[jx0 * (m + 1) + ix0] = sqrt((x - ix0 * h) * (x - ix0 * h) + (y - jx0 * h) * (y - jx0 * h)) * f[jx0 * (m + 1) + ix0];
  u[jx0 * (m + 1) + ix1] = sqrt((x - ix1 * h) * (x - ix1 * h) + (y - jx0 * h) * (y - jx0 * h)) * f[jx0 * (m + 1) + ix1];
  u[jx1 * (m + 1) + ix0] = sqrt((x - ix0 * h) * (x - ix0 * h) + (y - jx1 * h) * (y - jx1 * h)) * f[jx1 * (m + 1) + ix0];
  u[jx1 * (m + 1) + ix1] = sqrt((x - ix1 * h) * (x - ix1 * h) + (y - jx1 * h) * (y - jx1 * h)) * f[jx1 * (m + 1) + ix1];
  
  std::vector<int> I, J, iI, iJ;
  for (int i = 0; i < m + 1; i++)
  {
    I.push_back(i);
    iI.push_back(m - i);
  }
  for (int i = 0; i < n + 1; i++)
  {
    J.push_back(i);
    iJ.push_back(n - i);
  }

  Eigen::VectorXd uvec_old = Eigen::Map<const Eigen::VectorXd>(u, (m + 1) * (n + 1)), uvec;
  bool converged = false;
  for (int i = 0; i < 100; i++)
  {
    // sweep(u, I, J, f, m, n, h, ix, jx);
    // sweep(u, iI, J, f, m, n, h, ix, jx);
    // sweep(u, iI, iJ, f, m, n, h, ix, jx);
    // sweep(u, I, iJ, f, m, n, h, ix, jx);
    sweep(u, I, J, f, m, n, h);
    sweep(u, iI, J, f, m, n, h);
    sweep(u, iI, iJ, f, m, n, h);
    sweep(u, I, iJ, f, m, n, h);
    uvec = Eigen::Map<const Eigen::VectorXd>(u, (m + 1) * (n + 1));
    double err = (uvec - uvec_old).norm() / uvec_old.norm();
    // printf("ERROR AT ITER %d: %g\n", i, err);
    if (err < 1e-8)
    {
      converged = true;
      break;
    }
    uvec_old = uvec;
  }

  if (!converged)
  {
    printf("ERROR: Eikonal does not converge!\n");
  }
}

// void backward(
//     double *grad_f,
//     const double *grad_u,
//     const double *u, const double *f, int m, int n, double h, int ix, int jx)
void backward(
    double *grad_f,
    const double *grad_u,
    const double *u, const double *f, int m, int n, double h, double x, double y)
{
  int ix0 = std::max(0, std::min((int)floor(x / h), m - 1));
  int jx0 = std::max(0, std::min((int)floor(y / h), n - 1));
  int ix1 = ix0 + 1;
  int jx1 = jx0 + 1;
  

  Eigen::VectorXd dFdf((m + 1) * (n + 1));
  for (int i = 0; i < (m + 1) * (n + 1); i++)
  {
    dFdf[i] = -2 * f[i] * h * h;
  }
  // dFdf[jx * (m + 1) + ix] = 0.0;
  dFdf[jx0 * (m + 1) + ix0] = -sqrt((x - ix0 * h) * (x - ix0 * h) + (y - jx0 * h) * (y - jx0 * h));
  dFdf[jx0 * (m + 1) + ix1] = -sqrt((x - ix1 * h) * (x - ix1 * h) + (y - jx0 * h) * (y - jx0 * h));
  dFdf[jx1 * (m + 1) + ix0] = -sqrt((x - ix0 * h) * (x - ix0 * h) + (y - jx1 * h) * (y - jx1 * h));
  dFdf[jx1 * (m + 1) + ix1] = -sqrt((x - ix1 * h) * (x - ix1 * h) + (y - jx1 * h) * (y - jx1 * h));

  std::vector<T> triplets;

  for (int j = 0; j < n + 1; j++)
  {
    for (int i = 0; i < m + 1; i++)
    {
      int idx = j * (m + 1) + i;
      // if (i == ix && j == jx)
      // {
      //   triplets.push_back(T(idx, idx, 1.0));
      //   continue;
      // }
      if ((i == ix0 && j == jx0) || (i == ix1 && j == jx0) || (i == ix0 && j == jx1) || (i == ix1 && j == jx1)) {
        triplets.push_back(T(idx, idx, 1.0));
        continue;
      }

      // double val = 0.0;
      if (i == 0)
      {
        if (u[idx] > u[j * (m + 1) + 1])
        {
          triplets.push_back(T(idx, idx, 2 * (u[idx] - u[j * (m + 1) + 1])));
          triplets.push_back(T(idx, j * (m + 1) + 1, 2 * (u[j * (m + 1) + 1] - u[idx])));

          // val += (u[idx]-u[j*(m+1)+1])*(u[idx]-u[j*(m+1)+1]);
        }
      }
      else if (i == m)
      {

        if (u[idx] > u[j * (m + 1) + m - 1])
        {
          triplets.push_back(T(idx, idx, 2 * (u[idx] - u[j * (m + 1) + m - 1])));
          triplets.push_back(T(idx, j * (m + 1) + m - 1, 2 * (u[j * (m + 1) + m - 1] - u[idx])));

          // val += (u[idx]-u[j*(m+1)+m-1])*(u[idx]-u[j*(m+1)+m-1]);
        }
      }
      else
      {

        double a = u[j * (m + 1) + i + 1] > u[j * (m + 1) + i - 1] ? u[j * (m + 1) + i - 1] : u[j * (m + 1) + i + 1];
        if (u[idx] > a)
        {
          triplets.push_back(T(idx, idx, 2 * (u[idx] - a)));
          if (u[j * (m + 1) + i + 1] > u[j * (m + 1) + i - 1])
            triplets.push_back(T(idx, j * (m + 1) + i - 1, 2 * (a - u[idx])));
          else
            triplets.push_back(T(idx, j * (m + 1) + i + 1, 2 * (a - u[idx])));

          // val += (a-u[idx])*(a-u[idx]);
        }
      }

      if (j == 0)
      {
        if (u[idx] > u[m + 1 + i])
        {
          triplets.push_back(T(idx, idx, 2 * (u[idx] - u[m + 1 + i])));
          triplets.push_back(T(idx, (m + 1) + i, 2 * (u[m + 1 + i] - u[idx])));

          // val += (u[idx]-u[m+1+i])*(u[idx]-u[m+1+i]);
        }
      }
      else if (j == n)
      {

        if (u[idx] > u[(n - 1) * (m + 1) + i])
        {
          triplets.push_back(T(idx, idx, 2 * (u[idx] - u[(n - 1) * (m + 1) + i])));
          triplets.push_back(T(idx, (n - 1) * (m + 1) + i, 2 * (u[(n - 1) * (m + 1) + i] - u[idx])));

          // val += (u[idx]-u[(n-1)*(m+1)+i])*(u[idx]-u[(n-1)*(m+1)+i]);
        }
      }
      else
      {
        double b = u[(j + 1) * (m + 1) + i] > u[(j - 1) * (m + 1) + i] ? u[(j - 1) * (m + 1) + i] : u[(j + 1) * (m + 1) + i];
        if (u[idx] > b)
        {
          triplets.push_back(T(idx, idx, 2 * (u[idx] - b)));
          if (u[(j + 1) * (m + 1) + i] > u[(j - 1) * (m + 1) + i])
            triplets.push_back(T(idx, (j - 1) * (m + 1) + i, 2 * (b - u[idx])));
          else
            triplets.push_back(T(idx, (j + 1) * (m + 1) + i, 2 * (b - u[idx])));

          // val += (b-u[idx])*(b-u[idx]);
        }
      }

      // val -= f[idx]*f[idx]*h*h;
      // printf("VAL = %g\n", val);
    }
  }

  SpMat A((m + 1) * (n + 1), (m + 1) * (n + 1));
  A.setFromTriplets(triplets.begin(), triplets.end());
  A = A.transpose();
  Eigen::SparseLU<SpMat> solver;

  Eigen::VectorXd g = Eigen::Map<const Eigen::VectorXd>(grad_u, (m + 1) * (n + 1));
  solver.analyzePattern(A);
  solver.factorize(A);
  Eigen::VectorXd res = solver.solve(g);
  for (int i = 0; i < (m + 1) * (n + 1); i++)
  {
    grad_f[i] = -res[i] * dFdf[i];
  }
}

// PyTorch extension interface
// torch::Tensor eikonal_forward(torch::Tensor f, double h, int ix, int jx) {
torch::Tensor eikonal_forward(torch::Tensor f, double h, double x, double y) {
    auto m = f.size(0) - 1;
    auto n = f.size(1) - 1;
    
    auto u = torch::zeros_like(f);
    
    // forward(u.data_ptr<double>(), f.data_ptr<double>(), m, n, h, ix, jx);
    forward(u.data_ptr<double>(), f.data_ptr<double>(), m, n, h, x, y);
    
    return u;
}

// torch::Tensor eikonal_backward(torch::Tensor grad_u, torch::Tensor u, torch::Tensor f, double h, int ix, int jx) {
torch::Tensor eikonal_backward(torch::Tensor grad_u, torch::Tensor u, torch::Tensor f, double h, double x, double y) {
    auto m = f.size(0) - 1;
    auto n = f.size(1) - 1;
    
    auto grad_f = torch::zeros_like(f);
    
    // backward(grad_f.data_ptr<double>(), grad_u.data_ptr<double>(), u.data_ptr<double>(), f.data_ptr<double>(), m, n, h, ix, jx);
    backward(grad_f.data_ptr<double>(), grad_u.data_ptr<double>(), u.data_ptr<double>(), f.data_ptr<double>(), m, n, h, x, y);
    
    return grad_f;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &eikonal_forward, "Eikonal2D forward");
    m.def("backward", &eikonal_backward, "Eikonal2D backward");
}