#include "00_main.h"

[[cpp11::register]] doubles_matrix<> group_sums_(const doubles_matrix<>& M,
                                                 const doubles_matrix<>& w,
                                                 const list& jlist) {
  // Auxiliary variables (fixed)
  const int J = jlist.size();
  const int P = M.ncol();

  // Auxiliary variables (storage)
  double denom;
  int i, j, p, I;
  writable::doubles_matrix<> num(P, 1);

  // Compute sum of weighted group sums
  for (j = 0; j < J; j++) {
    // Subset j-th group
    integers indexes = as_cpp<integers>(jlist[j]);
    I = indexes.size();

    // Compute numerator of the weighted group sum
    writable::doubles_matrix<> num(P, 1);
    for (p = 0; p < P; p++) {
      num(p, 0) = 0.0;
      for (i = 0; i < I; i++) {
        num(p, 0) += M(indexes[i], p);
      }
    }

    // Compute denominator of the weighted group sum
    denom = 0.0;
    for (i = 0; i < I; i++) {
      denom += w(indexes[i], 0);
    }

    // Add / denom;
    for (p = 0; p < P; p++) {
      num(p, 0) /= denom;
    }
  }

  // Return vector
  return num;
}

[[cpp11::register]] doubles_matrix<> group_sums_spectral_(
    const doubles_matrix<>& M, const doubles_matrix<>& v,
    const doubles_matrix<>& w, const int K, const list& jlist) {
  // Auxiliary variables (fixed)
  const int J = jlist.size();
  const int P = M.ncol();

  // Auxiliary variables (storage)
  double denom;
  int i, j, k, p, I;
  writable::doubles_matrix<> num(P, 1);

  // Compute sum of weighted group sums
  for (j = 0; j < J; j++) {
    // Subset j-th group
    integers indexes = as_cpp<integers>(jlist[j]);
    I = indexes.size();

    // Compute numerator of the weighted group sum given bandwidth 'L'
    for (p = 0; p < P; p++) {
      num(p, 0) = 0.0;
      for (k = 1; k <= K; k++) {
        for (i = k; i < I; i++) {
          num(p, 0) += M(indexes[i], p) * v(indexes[i - k], 0) * I / (I - k);
        }
      }
    }

    // Compute denominator of the weighted group sum
    denom = 0.0;
    for (i = 0; i < I; i++) {
      denom += w(indexes[i], 0);
    }

    // Add weighted group sum
    for (p = 0; p < P; p++) {
      num(p, 0) /= denom;
    }
  }

  // Return vector
  return num;
}

[[cpp11::register]] doubles_matrix<> group_sums_var_(const doubles_matrix<>& M,
                                                     const list& jlist) {
  // Auxiliary variables (fixed)
  const int J = jlist.size();
  const int P = M.ncol();

  // Auxiliary variables (storage)
  int i, j, p, q, I;
  writable::doubles_matrix<> v(P, 1);
  writable::doubles_matrix<> V(P, P);

  // Compute covariance matrix
  for (p = 0; p < P; p++) {
    for (q = 0; q < P; q++) {
      V(p, q) = 0.0;
    }
  }

  for (j = 0; j < J; j++) {
    // Subset j-th group
    integers indexes = as_cpp<integers>(jlist[j]);
    I = indexes.size();

    // Compute group sum
    for (p = 0; p < P; p++) {
      v(p, 0) = 0.0;
      for (i = 0; i < I; ++i) {
        v(p, 0) += M(indexes[i], p);
      }
    }

    // Add to covariance matrix
    for (p = 0; p < P; p++) {
      for (q = 0; q < P; q++) {
        V(p, q) += v(p, 1) * v(q, 1);
      }
    }
  }

  // Return matrix
  return V;
}

[[cpp11::register]] doubles_matrix<> group_sums_cov_(const doubles_matrix<>& M,
                                                     const doubles_matrix<>& N,
                                                     const list& jlist) {
  // Auxiliary variables (fixed)
  const int J = jlist.size();
  const int P = M.ncol();

  // Auxiliary variables (storage)
  int i, j, p, q, s, I;
  writable::doubles_matrix<> V(P, P);

  // Compute covariance matrix
  for (p = 0; p < P; ++p) {
    for (q = 0; q < P; ++q) {
      V(p, q) = 0.0;
    }
  }

  for (j = 0; j < J; ++j) {
    // Subset j-th group
    integers indexes = as_cpp<integers>(jlist[j]);
    I = indexes.size();

    // Add to covariance matrix
    for (p = 0; p < P; ++p) {
      for (q = 0; q < P; ++q) {
        for (i = 0; i < I; ++i) {
          for (s = i + 1; s < I; ++s) {
            V(q, p) += M(indexes[i], q) * N(indexes[s], p);
          }
        }
      }
    }
  }

  // Return matrix
  return V;
}
