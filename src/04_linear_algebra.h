#ifndef CAPYBARA_LINALG_H
#define CAPYBARA_LINALG_H

// Weighted or unweighted crossproduct (X'WX or X'X)
inline mat crossproduct(const mat &X, const vec &w, crossproduct_results &ws,
                        bool use_weights) {
  if (use_weights) {
    // For large n and small p, use explicit loop for cache efficiency
    if (X.n_rows > 1000 && X.n_cols < 50) {
      mat XtWX(X.n_cols, X.n_cols, fill::zeros);

      for (uword i = 0; i < X.n_rows; i++) {
        const rowvec xi = X.row(i);
        XtWX += (w(i) * xi.t()) * xi; // Outer product for each row
      }
      return XtWX;
    } else {
      // For other cases, use weighted multiplication
      const vec sqrt_w = sqrt(w);
      const mat Xw = X.each_col() % sqrt_w;
      return Xw.t() * Xw;
    }
  }
  // Unweighted crossproduct
  return X.t() * X;
}

#endif // CAPYBARA_LINALG_H
