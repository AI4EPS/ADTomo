#include <torch/torch.h>
#include <torch/extension.h>

#include <vector>
#include <algorithm>
#include "Eigen/Core"
#include "Eigen/SparseCore"
#include "Eigen/SparseLU"
#include <vector>
#include <iostream>
#include <utility>  
typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> T;


double solution(double a, double b, double f, double h){
  double d = fabs(a-b);
  if(d>=f*h) 
    return std::min(a, b) + f*h;
  else 
    return (a+b+sqrt(2*f*f*h*h-(a-b)*(a-b)))/2;
}

void sweep(torch::Tensor &u, 
    const std::vector<int>& I, const std::vector<int>& J,
    const torch::Tensor &f, int m, int n, double h, int ix, int jx){

    auto uval = u.accessor<double,1>();
    auto fval = f.accessor<double,1>();

    for(int i: I){
      for(int j:J){
        if (i==ix && j==jx) continue; 
        double a, b;
        if (i==0){
          a = uval[j*(m+1)+1];
        }
        else if (i==m){
          a = uval[j*(m+1)+m-1];
        }
        else{
          a = std::min(uval[j*(m+1)+i+1], uval[j*(m+1)+i-1]);
        }
        if (j==0){
          b = uval[(m+1)+i];
        }
        else if (j==n){
          b = uval[(n-1)*(m+1)+i];
        }
        else{
          b = std::min(uval[(j-1)*(m+1)+i], uval[(j+1)*(m+1)+i]);
        }
        double u_new = solution(a, b, fval[j*(m+1)+i], h);
        uval[j*(m+1)+i] = std::min(uval[j*(m+1)+i],u_new);
      }
    }
    
}

void forward(torch::Tensor &u, const torch::Tensor &f, int m, int n, double h, int ix, int jx){

  m=m-1;
  n=n-1;

  auto uval = u.accessor<double,1>();
  double *udat = u.data_ptr<double>();
  auto fval = f.accessor<double,1>();

  for(int i=0;i<m+1;i++){
    for(int j=0;j<n+1;j++){
      uval[j*(m+1)+i] = 100000.0;
      if (i==ix && j==jx) uval[j*(m+1)+i] = 0.0;
    }
  }

  std::vector<int> I, J, iI, iJ;
  for(int i=0;i<m+1;i++) {
    I.push_back(i);
    iI.push_back(m-i);
  }
  for(int i=0;i<n+1;i++) {
    J.push_back(i);
    iJ.push_back(n-i);
  }

  Eigen::VectorXd uvec_old = Eigen::Map<const Eigen::VectorXd>(udat, (m+1)*(n+1)), uvec;
  bool converged = false;
  for(int i = 0;i<100;i++){
    sweep(u, I, J, f, m, n, h, ix, jx);
    sweep(u, iI, J, f, m, n, h, ix, jx);
    sweep(u, iI, iJ, f, m, n, h, ix, jx);
    sweep(u, I, iJ, f, m, n, h, ix, jx);
    uvec = Eigen::Map<const Eigen::VectorXd>(udat, (m+1)*(n+1));
    double err = (uvec-uvec_old).norm()/uvec_old.norm();
    if (err < 1e-8){ 
      converged = true;
      break; 
    }
    uvec_old = uvec;
  }

  if(!converged){
    printf("ERROR: Eikonal does not converge!\n");
  }
  

}

void backward(
  torch::Tensor &grad_f, 
  torch::Tensor &grad_u,
  const torch::Tensor &u, const torch::Tensor &f, int m, int n, double h, int ix, int jx){

    m=m-1;
    n=n-1;

    auto fval = f.accessor<double,1>();
    auto uval = u.accessor<double,1>();
    auto grad_fval = grad_f.accessor<double,1>();
    double *grad_udat = grad_u.data_ptr<double>();

    Eigen::VectorXd dFdf((m+1)*(n+1));
    for (int i=0;i<(m+1)*(n+1);i++){
      dFdf[i] = -2*fval[i]*h*h;
    }
    dFdf[jx*(m+1)+ix] = 0.0; 
    double val = 0.0;
    std::vector<T> triplets;

    for(int j=0;j<n+1;j++){
      for(int i=0;i<m+1;i++){
        int idx = j*(m+1)+i;
        if (i==ix && j==jx) {
          triplets.push_back(T(idx, idx, 1.0));
          continue;
        }

        if(i==0){
          if(uval[idx]>uval[j*(m+1)+1]){
            triplets.push_back(T(idx, idx, 2*(uval[idx]-uval[j*(m+1)+1]) ));
            triplets.push_back(T(idx, j*(m+1)+1, 2*(uval[j*(m+1)+1]-uval[idx]) ));

          val += (uval[idx]-uval[j*(m+1)+1])*(uval[idx]-uval[j*(m+1)+1]);
          }
        }
        else if (i==m){

          if(uval[idx]>uval[j*(m+1)+m-1]){
            triplets.push_back(T(idx, idx, 2*(uval[idx]-uval[j*(m+1)+m-1]) ));
            triplets.push_back(T(idx, j*(m+1)+m-1, 2*(uval[j*(m+1)+m-1]-uval[idx]) ));

          val += (uval[idx]-uval[j*(m+1)+m-1])*(uval[idx]-uval[j*(m+1)+m-1]);
          }

        }
        else {

          double a = uval[j*(m+1)+i+1]>uval[j*(m+1)+i-1] ? uval[j*(m+1)+i-1] : uval[j*(m+1)+i+1];
          if (uval[idx]>a){
            triplets.push_back(T(idx, idx, 2*(uval[idx]-a) ));
            if (uval[j*(m+1)+i+1]>uval[j*(m+1)+i-1])
              triplets.push_back(T(idx, j*(m+1)+i-1, 2*(a-uval[idx]) ));
            else
              triplets.push_back(T(idx, j*(m+1)+i+1, 2*(a-uval[idx]) ));

            val += (a-uval[idx])*(a-uval[idx]);
          }

        }

        if (j==0){
          if (uval[idx]>uval[m+1+i]){
            triplets.push_back(T(idx, idx, 2*(uval[idx]-uval[m+1+i]) ));
            triplets.push_back(T(idx, (m+1)+i, 2*(uval[m+1+i]-uval[idx]) ));

          val += (uval[idx]-uval[m+1+i])*(uval[idx]-uval[m+1+i]);
          }

        }
        else if(j==n){

          if (uval[idx]>uval[(n-1)*(m+1)+i]){
            triplets.push_back(T(idx, idx, 2*(uval[idx]-uval[(n-1)*(m+1)+i]) ));
            triplets.push_back(T(idx, (n-1)*(m+1)+i, 2*(uval[(n-1)*(m+1)+i]-uval[idx]) ));

          val += (uval[idx]-uval[(n-1)*(m+1)+i])*(uval[idx]-uval[(n-1)*(m+1)+i]);
          }

        }
        else {
          double b = uval[(j+1)*(m+1)+i]>uval[(j-1)*(m+1)+i] ? uval[(j-1)*(m+1)+i] : uval[(j+1)*(m+1)+i];
          if (uval[idx]>b){
            triplets.push_back(T(idx, idx, 2*(uval[idx]-b) ));
            if (uval[(j+1)*(m+1)+i]>uval[(j-1)*(m+1)+i])
              triplets.push_back(T(idx, (j-1)*(m+1)+i, 2*(b-uval[idx]) ));
            else
              triplets.push_back(T(idx, (j+1)*(m+1)+i, 2*(b-uval[idx]) ));

            val += (b-uval[idx])*(b-uval[idx]);
          }
        }

        val -= fval[idx]*fval[idx]*h*h;
        
      }
    }


    SpMat A((m+1)*(n+1), (m+1)*(n+1));
    A.setFromTriplets(triplets.begin(), triplets.end());
    A = A.transpose();
    Eigen::SparseLU<SpMat> solver;
     
    Eigen::VectorXd g = Eigen::Map<const Eigen::VectorXd>(grad_udat, (m+1)*(n+1));
    solver.analyzePattern(A);

    solver.factorize(A);

    Eigen::VectorXd res = solver.solve(g);

    for(int i=0;i<(m+1)*(n+1);i++){
      grad_fval[i] = -res[i] * dFdf[i];
    }
    
}

// pybind module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "EIK2D forward");
  m.def("backward", &backward, "EIK2D backward");
}

