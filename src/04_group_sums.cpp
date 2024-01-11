#include "00_main.hpp"
#include "01_cpp11_to_from_arma.hpp"

Col<double> group_sums_vec_(const Mat<double>& M, const Col<double>& w,
                            const list& jlist) {
  // Auxiliary variables (fixed)
  const int J = jlist.size();
  const int P = M.n_cols;
  
  // Auxiliary variables (storage)
  double denom;
  int j, p, t, T;
  Col<double> b(P);
  Col<double> num(P);
  
  // Compute sum of weighted group sums
  b.zeros();
  for (j = 0 ; j < J ; ++j) {
    // Subset j-th group
    integers indexes = as_cpp<integers>(jlist[j]);
    T = indexes.size();
    
    // Compute numerator of the weighted group sum
    num.zeros();
    for (p = 0 ; p < P ; ++p) {
      for (t = 0 ; t < T ; ++t) {
        num(p) += M(indexes[t], p);
      }
    }
    
    // Compute denominator of the weighted group sum
    denom = 0.0;
    for (t = 0 ; t < T ; ++t) {
      denom += w(indexes[t]);
    }
    
    // Add weighted group sum
    b += num / denom;
  }
  
  // Return vector
  return b;
}

Col<double> group_sums_spectral_vec_(const Mat<double>& M, const Col<double>& v,
                                     const Col<double>& w, const int L,
                                     const list& jlist) {
  // Auxiliary variables (fixed)
  const int J = jlist.size();
  const int P = M.n_cols;
  
  // Auxiliary variables (storage)
  double denom;
  int j, l, p, t, T;
  Col<double> b(P);
  Col<double> num(P);
  
  // Compute sum of weighted group sums
  b.zeros();
  for (j = 0 ; j < J ; ++j) {
    // Subset j-th group
    integers indexes = as_cpp<integers>(jlist[j]);
    T = indexes.size();
    
    // Compute numerator of the weighted group sum given bandwidth 'L'
    num.zeros();
    for (p = 0 ; p < P ; ++p) {
      for (l = 1 ; l <= L ; ++l) {
        for (t = l ; t < T ; ++t) {
          num(p) += M(indexes[t], p) * v(indexes[t - l]) * T / (T - l);
        }
      }
    }
    
    // Compute denominator of the weighted group sum
    denom = 0.0;
    for (t = 0 ; t < T ; ++t) {
      denom += w(indexes[t]);
    }
    
    // Add weighted group sum
    b += num / denom;
  }
  
  // Return vector
  return b;
}

Mat<double> group_sums_var_mat_(
    const Mat<double>& M,
    const list& jlist
  ) {
  // Auxiliary variables (fixed)
  const int J = jlist.size();
  const int P = M.n_cols;
  
  // Auxiliary variables (storage)
  int j, p, t, T;
  Col<double> v(P);
  Mat<double> V(P, P);
  
  // Compute covariance matrix
  V.zeros();
  for (j = 0 ; j < J ; ++j) {
    // Subset j-th group
    integers indexes = as_cpp<integers>(jlist[j]);
    T = indexes.size();
    
    // Compute group sum
    v.zeros();
    for (p = 0 ; p < P ; ++p) {
      for (t = 0 ; t < T ; ++t) {
        v(p) += M(indexes[t], p);
      }
    }
    
    // Add to covariance matrix
    V += v * v.t();
  }
  
  // Return matrix 
  return V;
}

Mat<double> group_sums_cov_mat_(const Mat<double>& M, const Mat<double>& N,
                        const list& jlist) {
  // Auxiliary variables (fixed)
  const int J = jlist.size();
  const int P = M.n_cols;
  
  // Auxiliary variables (storage)
  int j, p, q, t, s, T;
  Mat<double> V(P, P);
  
  // Compute covariance matrix
  V.zeros();
  for (j = 0 ; j < J ; ++j) {
    // Subset j-th group
    integers indexes = as_cpp<integers>(jlist[j]);
    T = indexes.size();
    
    // Add to covariance matrix
    for (p = 0 ; p < P ; ++p) {
      for (q = 0 ; q < P ; ++q) {
        for (t = 0 ; t < T ; ++t) {
          for (s = t + 1 ; s < T ; ++s) {
            V(q, p) += M(indexes[t], q) * N(indexes[s], p);
          }
        }
      }
    }
  }
  
  // Return matrix 
  return V;
}

// Now I have to write functions with outputs that R understands

[[cpp11::register]] doubles_matrix<> group_sums_(const doubles_matrix<>& M0,
                    const doubles& w0,
                    const list& jlist) {
  // Cast to Armadillo types
  Mat<double> M = doubles_matrix_to_Mat_(M0);
  Col<double> w = doubles_to_Vec_(w0);

  Col<double> b0 = group_sums_vec_(M, w, jlist);

  doubles_matrix<> B = Vec_to_doubles_matrix_(b0);

  return B;
}

[[cpp11::register]] doubles_matrix<> group_sums_spectral_(
    const doubles_matrix<>& M0, const doubles& v0, const doubles& w0,
    const int L, const list& jlist) {
  // Cast to Armadillo types
  Mat<double> M = doubles_matrix_to_Mat_(M0);
  Col<double> v = doubles_to_Vec_(v0);
  Col<double> w = doubles_to_Vec_(w0);

  Col<double> b0 = group_sums_spectral_vec_(M, v, w, L, jlist);

  doubles_matrix<> B = Vec_to_doubles_matrix_(b0);

  return B;
}

[[cpp11::register]] doubles_matrix<> group_sums_var_(const doubles_matrix<>& M0,
                                                     const list& jlist) {
  // Cast to Armadillo types
  Mat<double> M = doubles_matrix_to_Mat_(M0);

  Mat<double> V0 = group_sums_var_mat_(M, jlist);

  doubles_matrix<> V = Mat_to_doubles_matrix(V0);

  return V;
}

[[cpp11::register]] doubles_matrix<> group_sums_cov_(const doubles_matrix<>& M0,
                                                     const doubles_matrix<>& N0,
                                                     const list& jlist) {
  // Cast to Armadillo types
  Mat<double> M = doubles_matrix_to_Mat_(M0);
  Mat<double> N = doubles_matrix_to_Mat_(N0);

  Mat<double> V0 = group_sums_cov_mat_(M, N, jlist);

  doubles_matrix<> V = Mat_to_doubles_matrix(V0);

  return V;
}
