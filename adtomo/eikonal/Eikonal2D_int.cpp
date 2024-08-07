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

typedef Eigen::SparseMatrix<double> SpMat;
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
           const double *f, int m, int n, double h, int ix, int jx)
{
  for (int i : I)
  {
    for (int j : J)
    {
      if (i == ix && j == jx)
        continue;
      double a, b;
      if (i == 0)
      {
        a = u[(i + 1) * (n + 1) + j];
      }
      else if (i == m)
      {
        a = u[(i - 1) * (n + 1) + j];
      }
      else
      {
        a = std::min(u[(i + 1) * (n + 1) + j], u[(i - 1) * (n + 1) + j]);
      }
      if (j == 0)
      {
        b = u[i * (n + 1) + 1];
      }
      else if (j == n)
      {
        b = u[i * (n + 1) + n - 1];
      }
      else
      {
        b = std::min(u[i * (n + 1) + j - 1], u[i * (n + 1) + j + 1]);
      }
      double u_new = solution(a, b, f[i * (n + 1) + j], h);
      u[i * (n + 1) + j] = std::min(u[i * (n + 1) + j], u_new);
    }
  }
}

void forward(double *u, const double *f, int m, int n, double h, int ix, int jx)
{
  for (int i = 0; i < m + 1; i++)
  {
    for (int j = 0; j < n + 1; j++)
    {
      u[i * (n + 1) + j] = 100000.0;
      if (i == ix && j == jx)
        u[i * (n + 1) + j] = 0.0;
    }
  }
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
    sweep(u, I, J, f, m, n, h, ix, jx);
    sweep(u, iI, J, f, m, n, h, ix, jx);
    sweep(u, iI, iJ, f, m, n, h, ix, jx);
    sweep(u, I, iJ, f, m, n, h, ix, jx);
    uvec = Eigen::Map<const Eigen::VectorXd>(u, (m + 1) * (n + 1));
    double err = (uvec - uvec_old).norm() / uvec_old.norm();
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

void backward(
    double *grad_f,
    const double *grad_u,
    const double *u, const double *f, int m, int n, double h, int ix, int jx)
{
  Eigen::VectorXd dFdf((m + 1) * (n + 1));
  for (int i = 0; i < (m + 1) * (n + 1); i++)
  {
    dFdf[i] = -2 * f[i] * h * h;
  }
  dFdf[ix * (n + 1) + jx] = 0.0;
  std::vector<T> triplets;

  for (int i = 0; i < m + 1; i++)
  {
    for (int j = 0; j < n + 1; j++)
    {
      int idx = i * (n + 1) + j;
      if (i == ix && j == jx)
      {
        triplets.push_back(T(idx, idx, 1.0));
        continue;
      }

      // double val = 0.0;
      if (i == 0)
      {
        if (u[idx] > u[(i + 1) * (n + 1) + j])
        {
          triplets.push_back(T(idx, idx, 2 * (u[idx] - u[(i + 1) * (n + 1) + j])));
          triplets.push_back(T(idx, (i + 1) * (n + 1) + j, 2 * (u[(i + 1) * (n + 1) + j] - u[idx])));

          // val += (u[idx]-u[j*(m+1)+1])*(u[idx]-u[j*(m+1)+1]);
        }
      }
      else if (i == m)
      {
        if (u[idx] > u[(i - 1) * (n + 1) + j])
        {
          triplets.push_back(T(idx, idx, 2 * (u[idx] - u[(i - 1) * (n + 1) + j])));
          triplets.push_back(T(idx, (i - 1) * (n + 1) + j, 2 * (u[(i - 1) * (n + 1) + j] - u[idx])));

          // val += (u[idx]-u[j*(m+1)+m-1])*(u[idx]-u[j*(m+1)+m-1]);
        }
      }
      else
      {
        double a = u[(i + 1) * (n + 1) + j] > u[(i - 1) * (n + 1) + j] ? u[(i - 1) * (n + 1) + j] : u[(i + 1) * (n + 1) + j];
        if (u[idx] > a)
        {
          triplets.push_back(T(idx, idx, 2 * (u[idx] - a)));
          if (u[(i + 1) * (n + 1) + j] > u[(i - 1) * (n + 1) + j])
            triplets.push_back(T(idx, (i - 1) * (n + 1) + j, 2 * (a - u[idx])));
          else
            triplets.push_back(T(idx, (i + 1) * (n + 1) + j, 2 * (a - u[idx])));

          // val += (a-u[idx])*(a-u[idx]);
        }
      }

      if (j == 0)
      {
        if (u[idx] > u[i * (n + 1) + 1])
        {
          triplets.push_back(T(idx, idx, 2 * (u[idx] - u[i * (n + 1) + 1])));
          triplets.push_back(T(idx, i * (n + 1) + 1, 2 * (u[i * (n + 1) + 1] - u[idx])));

          // val += (u[idx]-u[m+1+i])*(u[idx]-u[m+1+i]);
        }
      }
      else if (j == n)
      {
        if (u[idx] > u[i * (n + 1) + n - 1])
        {
          triplets.push_back(T(idx, idx, 2 * (u[idx] - u[i * (n + 1) + n - 1])));
          triplets.push_back(T(idx, i * (n + 1) + n - 1, 2 * (u[i * (n + 1) + n - 1] - u[idx])));

          // val += (u[idx]-u[(n-1)*(m+1)+i])*(u[idx]-u[(n-1)*(m+1)+i]);
        }
      }
      else
      {
        double b = u[i * (n + 1) + j + 1] > u[i * (n + 1) + j - 1] ? u[i * (n + 1) + j - 1] : u[i * (n + 1) + j + 1];
        if (u[idx] > b)
        {
          triplets.push_back(T(idx, idx, 2 * (u[idx] - b)));
          if (u[i * (n + 1) + j + 1] > u[i * (n + 1) + j - 1])
            triplets.push_back(T(idx, i * (n + 1) + j - 1, 2 * (b - u[idx])));
          else
            triplets.push_back(T(idx, i * (n + 1) + j + 1, 2 * (b - u[idx])));

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
torch::Tensor eikonal_forward(torch::Tensor f, double h, int ix, int jx) {
    auto m = f.size(0) - 1;
    auto n = f.size(1) - 1;
    
    auto u = torch::zeros_like(f);
    
    forward(u.data_ptr<double>(), f.data_ptr<double>(), m, n, h, ix, jx);
    
    return u;
}

torch::Tensor eikonal_backward(torch::Tensor grad_u, torch::Tensor u, torch::Tensor f, double h, int ix, int jx) {
    auto m = f.size(0) - 1;
    auto n = f.size(1) - 1;
    
    auto grad_f = torch::zeros_like(f);
    
    backward(grad_f.data_ptr<double>(), grad_u.data_ptr<double>(), u.data_ptr<double>(), f.data_ptr<double>(), m, n, h, ix, jx);
    
    return grad_f;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &eikonal_forward, "Eikonal2D forward");
    m.def("backward", &eikonal_backward, "Eikonal2D backward");
}