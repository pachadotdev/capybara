// fixed_effects_wrapper.cpp
#include "fixed_effects_impl.hpp"
#include <cpp11.hpp>
#include <cpp11/doubles.hpp>
#include <cpp11/integers.hpp>
#include <cpp11/list.hpp>
#include <cpp11/matrix.hpp>

using namespace cpp11;

// Helper to convert cpp11 matrix to arma mat
arma::mat cpp11_to_arma_mat(const doubles_matrix<> &m) {
  int nrows = m.nrow();
  int ncols = m.ncol();
  arma::mat result(nrows, ncols);

  for (int j = 0; j < ncols; ++j) {
    for (int i = 0; i < nrows; ++i) {
      result(i, j) = m(i, j);
    }
  }

  return result;
}

// Helper to convert cpp11 vector to arma vec
arma::vec cpp11_to_arma_vec(const doubles &v) {
  arma::vec result(v.size());
  for (int i = 0; i < v.size(); ++i) {
    result(i) = v[i];
  }
  return result;
}

// Helper to convert arma mat to cpp11 matrix
doubles_matrix<> arma_to_cpp11_mat(const arma::mat &m) {
  writable::doubles_matrix<> result(m.n_rows, m.n_cols);

  for (size_t j = 0; j < m.n_cols; ++j) {
    for (size_t i = 0; i < m.n_rows; ++i) {
      result(i, j) = m(i, j);
    }
  }

  return result;
}

// Convert list of matrices to vector of arma matrices
std::vector<arma::mat> list_to_arma_mat_vec(const list &l) {
  std::vector<arma::mat> result;
  result.reserve(l.size());

  for (int i = 0; i < l.size(); ++i) {
    if (TYPEOF(l[i]) == REALSXP) {
      doubles_matrix<> m(l[i]);
      result.push_back(cpp11_to_arma_mat(m));
    } else {
      result.push_back(arma::mat());
    }
  }

  return result;
}

// Convert list of integer vectors to vector of arma uvec (with 0-based
// indexing)
std::vector<arma::uvec> list_to_arma_uvec_vec(const list &l) {
  std::vector<arma::uvec> result;
  result.reserve(l.size());

  for (int i = 0; i < l.size(); ++i) {
    integers iv(l[i]);
    arma::uvec uv(iv.size());
    for (int j = 0; j < iv.size(); ++j) {
      uv(j) = iv[j] - 1; // Convert to 0-based indexing
    }
    result.push_back(uv);
  }

  return result;
}

// Main wrapper function
[[cpp11::register]] list cpp_demean_(SEXP y_sexp, SEXP X_sexp, doubles weights,
                                     int max_iter, double tol,
                                     integers n_levels, list fe_ids_list,
                                     integers slope_flags,
                                     list slope_vars_list) {

  // Convert inputs
  arma::mat y_mat, X_mat;
  arma::vec weights_vec;

  // Handle y input (can be matrix or vector)
  if (TYPEOF(y_sexp) == REALSXP) {
    SEXP dims = Rf_getAttrib(y_sexp, R_DimSymbol);
    if (Rf_isNull(dims)) {
      // It's a vector
      doubles y_vec(y_sexp);
      y_mat = arma::mat(y_vec.size(), 1);
      for (int i = 0; i < y_vec.size(); ++i) {
        y_mat(i, 0) = y_vec[i];
      }
    } else {
      // It's a matrix
      doubles_matrix<> y_cpp11(y_sexp);
      y_mat = cpp11_to_arma_mat(y_cpp11);
    }
  }

  // Handle X input (can be matrix or NULL)
  if (TYPEOF(X_sexp) == REALSXP) {
    SEXP dims = Rf_getAttrib(X_sexp, R_DimSymbol);
    if (!Rf_isNull(dims)) {
      doubles_matrix<> X_cpp11(X_sexp);
      X_mat = cpp11_to_arma_mat(X_cpp11);
    }
  }

  // Convert weights
  if (weights.size() > 1) {
    weights_vec = cpp11_to_arma_vec(weights);
  } else {
    weights_vec = arma::vec();
  }

  // Convert lists
  std::vector<int> n_levels_vec(n_levels.begin(), n_levels.end());
  std::vector<int> slope_flags_vec(slope_flags.begin(), slope_flags.end());
  std::vector<arma::uvec> fe_ids_vec = list_to_arma_uvec_vec(fe_ids_list);
  std::vector<arma::mat> slope_vars_vec = list_to_arma_mat_vec(slope_vars_list);

  // Call implementation
  DemeanResult result =
      demean_impl(y_mat, X_mat, weights_vec, max_iter, tol, n_levels_vec,
                  fe_ids_vec, slope_flags_vec, slope_vars_vec);

  // Convert results back to cpp11
  writable::list output;

  if (result.X_demean.n_elem > 0) {
    output.push_back(named_arg("X_demean") =
                         arma_to_cpp11_mat(result.X_demean));
  } else {
    output.push_back(named_arg("X_demean") = doubles_matrix<>(0, 0));
  }

  output.push_back(named_arg("y_demean") = arma_to_cpp11_mat(result.y_demean));

  writable::integers iterations(result.iterations.size());
  for (size_t i = 0; i < result.iterations.size(); ++i) {
    iterations[i] = result.iterations[i];
  }
  output.push_back(named_arg("iterations") = iterations);

  return output;
}

// Additional wrapper for checking NA/Inf values
[[cpp11::register]] list cpp_which_na_inf_(SEXP x, int nthreads) {
  bool any_na = false;
  bool any_inf = false;
  writable::logicals is_na_inf;

  if (TYPEOF(x) == REALSXP) {
    SEXP dims = Rf_getAttrib(x, R_DimSymbol);

    if (Rf_isNull(dims)) {
      // Vector
      doubles xvec(x);
      is_na_inf = writable::logicals(xvec.size());

      for (int i = 0; i < xvec.size(); ++i) {
        is_na_inf[i] = false;
        if (ISNAN(xvec[i])) {
          is_na_inf[i] = true;
          any_na = true;
        } else if (!R_FINITE(xvec[i])) {
          is_na_inf[i] = true;
          any_inf = true;
        }
      }
    } else {
      // Matrix
      doubles_matrix<> xmat(x);
      int nrows = xmat.nrow();
      is_na_inf = writable::logicals(nrows);

      for (int i = 0; i < nrows; ++i) {
        is_na_inf[i] = false;
        for (int j = 0; j < xmat.ncol(); ++j) {
          if (ISNAN(xmat(i, j))) {
            is_na_inf[i] = true;
            any_na = true;
            break;
          } else if (!R_FINITE(xmat(i, j))) {
            is_na_inf[i] = true;
            any_inf = true;
            break;
          }
        }
      }
    }
  }

  if (!any_na && !any_inf) {
    is_na_inf = writable::logicals(1);
    is_na_inf[0] = false;
  }

  writable::list result;
  result.push_back(named_arg("any_na") = any_na);
  result.push_back(named_arg("any_inf") = any_inf);
  result.push_back(named_arg("any_na_inf") = (any_na || any_inf));
  result.push_back(named_arg("is_na_inf") = is_na_inf);

  return result;
}
