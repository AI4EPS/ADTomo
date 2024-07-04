#include <torch/torch.h>
#include <vector>
#include <algorithm>
#include <tuple>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <fstream>
#include "../Eigen/Core"
#include "../Eigen/SparseCore"
#include "../Eigen/SparseLU"

#include <set>
typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;

#define u(i,j,k) u[(i) * n * l + (j) * l + (k)]
#define f(i,j,k) f[(i) * n * l + (j) * l + (k)]
#define get_id(i,j,k) ((i) * n * l + (j) * l + (k))

double calculate_unique_solution(double a1_, double a2_, 
        double a3_, double f, double h){
    double a1=a1_, a2=a2_, a3=a3_, temp;
    if(a1>a2) {temp = a1; a1 = a2; a2 = temp;}
    if(a1>a3) {temp = a1; a1 = a3; a3 = temp;}
    if(a2>a3) {temp = a2; a2 = a3; a3 = temp;}

    double x = a1 + f * h;
    if (x <= a2) return x;
    double B = - (a1 + a2);
    double C = (a1 * a1 + a2 * a2 - f * f * h * h)/2.0;
    x = (-B + sqrt(B * B - 4 * C))/2.0;
    if (x <= a3) return x;
    B = -2.0*(a1 + a2 + a3)/3.0;
    C = (a1 * a1 + a2 * a2 + a3 * a3 - f * f * h * h)/3.0;
    x = (-B + sqrt(B * B - 4 * C))/2.0;
    return x;
}

void sweeping_over_I_J_K(torch::Tensor &u, torch::Tensor &f,
        int dirI,
        int dirJ,
        int dirK,
        int m, int n, int l, double h){
        
    auto I = std::make_tuple(dirI==1?0:m-1, dirI==1?m:-1, dirI);
    auto J = std::make_tuple(dirJ==1?0:n-1, dirJ==1?n:-1, dirJ);
    auto K = std::make_tuple(dirK==1?0:l-1, dirK==1?l:-1, dirK);
    

    // accessor
    auto uval = u.accessor<double,1>();
    auto fval = f.accessor<double,1>();
    auto nl = n*l;

    for (int i = std::get<0>(I); i != std::get<1>(I); i += std::get<2>(I))
        for (int j = std::get<0>(J); j != std::get<1>(J); j += std::get<2>(J))
            for (int k = std::get<0>(K); k != std::get<1>(K); k += std::get<2>(K)){
                double uxmin = i==0 ? uval[((i+1)*n*l)+(j*l)+k]: \
                                (i==m-1 ? uval[((i-1)*n*l)+(j*l)+k] : std::min(uval[((i+1)*nl)+(j*l)+k], uval[((i-1)*nl)+(j*l)+k]));
                double uymin = j==0 ? uval[(i*nl)+((j+1)*l)+k] : \
                                (j==n-1 ? uval[(i*nl)+((j-1)*l)+k] : std::min(uval[(i*nl)+((j+1)*l)+k], uval[(i*nl)+((j-1)*l)+k]));
                double uzmin = k==0 ? uval[(i*nl)+(j*l)+(k+1)] : \
                                (k==l-1 ? uval[(i*nl)+(j*l)+(k-1)] : std::min(uval[(i*nl)+(j*l)+(k+1)], uval[(i*nl)+(j*l)+(k-1)]));
                double u_new = calculate_unique_solution(uxmin, uymin, uzmin, fval[(i*nl)+(j*l)+k], h);
                uval[(i*nl)+(j*l)+k] = std::min(u_new, uval[(i*nl)+(j*l)+k]);
            }

}

void sweeping(torch::Tensor &u, torch::Tensor &f,int m, int n, int l, double h){
    std::cout << "Sweeping" << std::endl;
    sweeping_over_I_J_K(u,f, 1, 1, 1, m, n, l, h);
    sweeping_over_I_J_K(u,f, -1, 1, 1, m, n, l, h);
    sweeping_over_I_J_K(u,f, -1, -1, 1, m, n, l, h);
    sweeping_over_I_J_K(u,f, 1, -1, 1, m, n, l, h);
    sweeping_over_I_J_K(u,f, 1, -1, -1, m, n, l, h);
    sweeping_over_I_J_K(u,f, 1, 1, -1, m, n, l, h);
    sweeping_over_I_J_K(u,f, -1, 1, -1, m, n, l, h);
    sweeping_over_I_J_K(u,f, -1, -1, -1, m, n, l, h);
}

void solve(torch::Tensor &u, torch::Tensor &f, double tol, bool verbose, int m, int n, int l, double h){
    // memcpy(u, u0, sizeof(double)*m*n*l);
    // copy data into tensor
    // u = u0.to(torch::kFloat64);
    // auto u_old = new double[m*n*l];


    std::cout << " Solving " << std::endl;
    // accessor for u
    auto uval = u.accessor<double,1>();

    for (int i = 0; i < 20; i++){
        // memcpy(u_old, u, sizeof(double)*m*n*l);
        torch::Tensor u_old = u.to(torch::kFloat64);
        auto u_oldval = u_old.accessor<double,1>();
        sweeping(u,f, m, n, l, h);
        double err = 0.0;

        for (int j = 0; j < m*n*l; j++){
            err = std::max(fabs(uval[j]-u_oldval[j]), err);
        }
        if (verbose){
            printf("Iteration %d, Error = %0.6e\n", i, err);
            std::cout << "error = " << err << std::endl;
        }
        if (err < tol) break;
        std::cout << "Iteration = " << i << std::endl;
    }
    // delete [] u_old;
    // delete u_old;
}

void backward(
            torch::Tensor &grad_u0, torch::Tensor &grad_f, const torch::Tensor &grad_u, 
            const torch::Tensor &u, const torch::Tensor &u0,  const torch::Tensor &f, double h, 
            int m, int n, int l){


    double *grad_udat = grad_u.data_ptr<double>();

    Eigen::VectorXd g(m*n*l);
    memcpy(g.data(), grad_udat, sizeof(double)*m*n*l);

    // accessor for u, u0
    auto uval = u.accessor<double,1>(); 
    auto u0val = u0.accessor<double,1>();
    auto grad_fval = grad_f.accessor<double,1>();
    auto grad_u0val = grad_u0.accessor<double,1>();
    auto fval = f.accessor<double,1>();
    auto nl = n*l;

    // calculate gradients for \partial L/\partial u0
    for (int i = 0; i < m*n*l; i++){
        // if (fabs(u[i] - u0[i])<1e-6) grad_u0[i] = g[i];
        if (uval[i] == u0val[i]) grad_u0val[i] = g[i];
        else grad_u0val[i] = 0.0;
    }

    // calculate gradients for \partial L/\partial f
    Eigen::VectorXd rhs(m*n*l);
    for (int i=0;i<m*n*l;i++){
      rhs[i] = -2*fval[i]*h*h;
    }

    std::vector<T> triplets;
    std::set<int> zero_id;
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            for (int k = 0; k < l; k++){

                int this_id = get_id(i, j, k);

                if (uval[this_id] == u0val[this_id]){
                    zero_id.insert(this_id);
                    g[this_id] = 0.0;
                    continue;
                }

                double uxmin = i==0 ? uval[((i+1)*nl)+(j*l)+k] : \
                                (i==m-1 ? uval[((i-1)*nl)+(j*l)+k] : std::min(uval[((i+1)*nl)+(j*l)+k], uval[((i-1)*nl)+(j*l)+k]));
                double uymin = j==0 ? uval[(i*nl)+((j+1)*l)+k] : \
                                (j==n-1 ? uval[(i*nl)+((j-1)*l)+k] : std::min(uval[(i*nl)+((j+1)*l)+k], uval[(i*nl)+((j-1)*l)+k]));
                double uzmin = k==0 ? uval[(i*nl)+(j*l)+(k+1)] : \
                                (k==l-1 ? uval[(i*nl)+(j*l)+(k-1)] : std::min(uval[(i*nl)+(j*l)+(k+1)], uval[(i*nl)+(j*l)+(k-1)]));

                int idx = i==0 ? get_id(i+1, j, k) : \
                                (i==m-1 ? get_id(i-1, j, k) : \
                                ( uval[((i+1)*nl)+(j*l)+k] > uval[((i-1)*nl)+(j*l)+k] ? get_id(i-1, j, k) : get_id(i+1, j, k)));
                int idy = j==0 ? get_id(i, j+1, k) : \
                                (j==n-1 ? get_id(i, j-1, k) : \
                                ( uval[(i*nl)+((j+1)*l)+k] > uval[(i*nl)+((j-1)*l)+k] ? get_id(i, j-1, k) : get_id(i, j+1, k)));
                int idz = k==0 ? get_id(i, j, k+1) : \
                                (k==l-1 ? get_id(i, j, k-1) : \
                                ( uval[(i*nl)+(j*l)+(k+1)] > uval[(i*nl)+(j*l)+(k-1)] ? get_id(i, j, k-1) : get_id(i, j, k+1)));

                bool this_id_is_not_zero = false;
                if (uval[(i*nl)+(j*l)+k] > uxmin){
                    this_id_is_not_zero = true;
                    triplets.push_back(T(this_id, this_id, 2.0 * (uval[(i*nl)+(j*l)+k] - uxmin)));
                    triplets.push_back(T(this_id, idx, -2.0 * (uval[(i*nl)+(j*l)+k] - uxmin)));
                }

                if (uval[(i*nl)+(j*l)+k] > uymin){
                    this_id_is_not_zero = true;
                    triplets.push_back(T(this_id, this_id, 2.0 * (uval[(i*nl)+(j*l)+k] - uymin)));
                    triplets.push_back(T(this_id, idy, -2.0 * (uval[(i*nl)+(j*l)+k] - uymin)));
                }

                if (uval[(i*nl)+(j*l)+k] > uzmin){
                    this_id_is_not_zero = true;
                    triplets.push_back(T(this_id, this_id, 2.0 * (uval[(i*nl)+(j*l)+k] - uzmin)));
                    triplets.push_back(T(this_id, idz, -2.0 * (uval[(i*nl)+(j*l)+k] - uzmin)));
                }

                if (!this_id_is_not_zero){
                    zero_id.insert(this_id);
                    g[this_id] = 0.0;
                }

            }
        }
    }

    if (zero_id.size()>0){
        for (auto& t : triplets){
            if (zero_id.count(t.col()) || zero_id.count(t.row())) t = T(t.col(), t.row(), 0.0);
        }
        for (auto idx: zero_id){
            triplets.push_back(T(idx, idx, 1.0));
        }
    }

    SpMat A(m*n*l, m*n*l);
    A.setFromTriplets(triplets.begin(), triplets.end());
    A = A.transpose();
    Eigen::SparseLU<SpMat> solver;
     
    solver.analyzePattern(A);
    solver.factorize(A);
    Eigen::VectorXd res = solver.solve(g);
    for(int i=0;i<m*n*l;i++){
      grad_fval[i] = -res[i] * rhs[i];
    }

}

// pybind module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &solve, "EIK3D forward");
  m.def("backward", &backward, "EIK3D backward");
}

