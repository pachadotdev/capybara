#ifndef CAPYBARA_LINALG_H
#define CAPYBARA_LINALG_H

inline mat crossproduct(const mat &X, const vec &w, crossproduct_results &ws,
                        bool use_weights) {
  if (use_weights) {
    // Avoid creating a full copy when we can compute directly
    if (X.n_rows > 1000 && X.n_cols < 50) {
      mat XtWX(X.n_cols, X.n_cols, fill::zeros);

      for (uword i = 0; i < X.n_rows; i++) {
        rowvec xi = X.row(i);
        xi *= w(i);
        XtWX += xi.t() * xi;
      }
      return XtWX;
    } else {
      // For other cases, use element-wise multiplication
      vec sqrt_w = sqrt(w);
      mat Xw = X.each_col() % sqrt_w;
      return Xw.t() * Xw;
    }
  }
  return X.t() * X;
}

#endif // CAPYBARA_LINALG_H
