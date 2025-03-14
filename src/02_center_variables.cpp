#include "00_main.h"

// Halperin method with aggressive precomputing and optimized memory use
void center_variables_(mat &V, const vec &w, const list &klist,
                       const double &tol, const size_t &maxiter,
                       const size_t &interrupt_iter) {
  // Auxiliary variables (fixed)
  const size_t N = V.n_rows;
  const size_t P = V.n_cols;
  const size_t K = klist.size();
  const double inv_sw = 1.0 / accu(w);

  // Auxiliary variables (storage)
  vec x(N), x0(N);
  field<field<uvec>> group_indices(K);
  field<vec> group_inverse_weights(K);

  // Precompute indices and inverse weights once
  for (size_t k = 0; k < K; ++k) {
    const list &jlist = klist[k];
    size_t J = jlist.size();

    field<uvec> indices(J);
    vec inverse_weights(J);

    for (size_t j = 0; j < J; ++j) {
      indices(j) = as_uvec(as_cpp<integers>(jlist[j]));
      inverse_weights(j) = 1.0 / accu(w.elem(indices(j)));
    }

    group_indices(k) = std::move(indices);
    group_inverse_weights(k) = std::move(inverse_weights);
  }

// Perform Halperin projections, parallelizing columns
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads) private(x, x0)
#endif
  for (size_t p = 0; p < P; ++p) {
    x = V.col(p);
    size_t interrupt = interrupt_iter;

    for (size_t iter = 0; iter < maxiter; ++iter) {
      if (iter == interrupt) {
        // Only main thread checks for interrupts
#ifdef _OPENMP
        if (omp_get_thread_num() == 0) {
          check_user_interrupt();
        }
#else
        check_user_interrupt();
#endif
        interrupt += interrupt_iter;
      }

      x0 = x;

      // Project onto group means
      for (size_t l = 0; l < K; ++l) {
        const size_t L = group_indices(l).size();
        for (size_t m = 0; m < L; ++m) {
          const uvec &coords = group_indices(l)(m);
          const double xbar =
              dot(w.elem(coords), x.elem(coords)) * group_inverse_weights(l)(m);
          x.elem(coords) -= xbar;
        }
      }

      // Check convergence (correct placement)
      double ratio = dot(abs(x - x0) / (1.0 + abs(x0)), w) * inv_sw;
      if (ratio < tol) {
        break;
      }
    }

    // Assign back at convergence
    V.col(p) = std::move(x);
  }
}

[[cpp11::register]] doubles_matrix<> center_variables_r_(
    const doubles_matrix<> &V_r, const doubles &w_r, const list &klist,
    const double &tol, const int &maxiter, const int &interrupt_iter) {
  mat V = as_Mat(V_r);
  vec w = as_Col(w_r);
  center_variables_(V, w, klist, tol, maxiter, interrupt_iter);
  return as_doubles_matrix(std::move(V));
}

// Kaczmarz demeaning
// void center_variables_(mat &V, const vec &w, const list &klist,
//                        const double &tol, const size_t &maxiter,
//                        const size_t &interrupt_iter) {
//   // Auxiliary variables (fixed)
//   const size_t P = V.n_cols;
//   const size_t K = klist.size();
//   const double inv_sw = 1.0 / accu(w);

//   // Auxiliary variables (storage)
//   size_t interrupt = static_cast<size_t>(interrupt_iter);
//   uvec coords;

//   // Precompute group indices and weights parallelizing over groups
//   field<field<uvec>> group_indices(K);
//   field<vec> group_inverse_weights(K);

// #ifdef _OPENMP
// #pragma omp parallel for schedule(static, n_threads)
// #endif
//   for (size_t k = 0; k < K; ++k) {
//     const list &jlist = klist[k];
//     size_t J = jlist.size();

//     field<uvec> indices(J);
//     vec inverse_weights(J);

//     for (size_t j = 0; j < J; ++j) {
//       indices(j) = as_uvec(as_cpp<integers>(jlist[j]));
//       inverse_weights(j) = 1.0 / accu(w.elem(indices(j)));
//     }

//     group_indices(k) = std::move(indices);
//     group_inverse_weights(k) = std::move(inverse_weights);
//   }

//   // Kaczmarz iterations parallelizing over columns
// #ifdef _OPENMP
// #pragma omp parallel for schedule(static, n_threads)
// #endif
//   for (size_t p = 0; p < P; ++p) {
//     for (size_t iter = 0; iter < maxiter; ++iter) {
//       if (iter == interrupt) {
//         check_user_interrupt();
//         interrupt += 1000;
//       }

//       vec x = V.col(p);
//       vec x0 = x;
//       double ratio;

//       for (size_t l = 0; l < K; ++l) {
//         size_t L = group_indices(l).size();
//         if (L == 0) continue;

//         for (size_t m = 0; m < L; ++m) {
//           const uvec &coords = group_indices(l)(m);
//           double xbar =
//               dot(w.elem(coords), x.elem(coords)) / accu(w.elem(coords));
//           x.elem(coords) -= xbar;
//         }
//       }

//       ratio = dot(abs(x - x0) / (1.0 + abs(x0)), w) * inv_sw;
//       if (ratio < tol) {
//         break;
//       }

//       V.col(p) = x;
//     }
//   }
// }
