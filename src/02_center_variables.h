#ifndef CAPYBARA_CENTER
#define CAPYBARA_CENTER

// Group mean subtraction
void center_variables_(mat &V, const vec &w, const list &klist,
                       const double &tol, const int &max_iter,
                       const int &iter_interrupt, const int &iter_ssr) {
  const size_t N = V.n_rows, P = V.n_cols, K = klist.size();
  const double inv_sw = 1.0 / accu(w);

  // Precompute group indices and weights
  field<field<uvec>> group_indices(K);
  field<vec> group_inv_w(K);
  for (size_t k = 0; k < K; ++k) {
    const list &jlist = klist[k];
    size_t J = jlist.size();
    field<uvec> idxs(J);
    vec invs(J);
    for (size_t j = 0; j < J; ++j) {
      idxs(j) = as_uvec(as_cpp<integers>(jlist[j]));
      invs(j) = 1.0 / accu(w.elem(idxs(j)));
    }
    group_indices(k) = std::move(idxs);
    group_inv_w(k) = std::move(invs);
  }

  for (size_t p = 0; p < P; ++p) {
    vec x = V.col(p);
    vec x0(N, fill::none);
    double ratio = 0.0;
    if (K == 2) {
      // Two-way FE specialization: alternate between FEs
      for (int iter = 0; iter < max_iter; ++iter) {
        x0 = x;
        for (size_t k = 0; k < 2; ++k) {
          const auto &idxs = group_indices(k);
          const auto &invs = group_inv_w(k);
          size_t L = idxs.n_elem;
          for (size_t l = 0; l < L; ++l) {
            const uvec &coords = idxs(l);
            if (coords.n_elem == 0)
              continue;
            double xbar = dot(w.elem(coords), x.elem(coords)) * invs(l);
            x.elem(coords) -= xbar;
          }
        }
        ratio = dot(abs(x - x0), w) * inv_sw;
        if (ratio < tol)
          break;
      }
    } else {
      // General k-way FE
      for (int iter = 0; iter < max_iter; ++iter) {
        x0 = x;
        for (size_t k = 0; k < K; ++k) {
          const auto &idxs = group_indices(k);
          const auto &invs = group_inv_w(k);
          size_t L = idxs.n_elem;
          for (size_t l = 0; l < L; ++l) {
            const uvec &coords = idxs(l);
            if (coords.n_elem == 0)
              continue;
            double xbar = dot(w.elem(coords), x.elem(coords)) * invs(l);
            x.elem(coords) -= xbar;
          }
        }
        ratio = dot(abs(x - x0), w) * inv_sw;
        if (ratio < tol)
          break;
      }
    }
    V.col(p) = x;
  }
}

#endif // CAPYBARA_CENTER
