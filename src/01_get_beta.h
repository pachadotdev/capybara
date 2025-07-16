#ifndef CAPYBARA_BETA
#define CAPYBARA_BETA

struct ModelResults {
  // Matrix computations
  mat XtX;
  vec XtY;
  mat decomp;
  vec work;
  mat Xt;
  mat Q;
  mat XW;

  // Core results
  vec coefficients;
  uvec coef_status;
  vec fitted_values;
  vec residuals;
  vec weights;
  mat hessian;
  // mat scores;
  bool success;

  // GLM-specific fields
  vec eta;                    // Linear predictor
  vec mu;                     // Mean values (response scale)
  double deviance;            // Current deviance
  double null_deviance;       // Null deviance
  size_t iter;               // Number of iterations
  bool conv;                 // Convergence status
  vec residuals_working;     // Working residuals
  vec residuals_response;    // Response residuals
  field<vec> fixed_effects;  // Recovered fixed effects
  bool has_fe;               // Whether fixed effects are present
  
  // GLM temporary workspace
  vec W;                     // IRLS weights
  vec W_tilde;              // Square root of IRLS weights
  vec Z;                    // Working dependent variable
  vec v_tilde;              // Weighted working variable
  mat X_tilde;              // Weighted design matrix
  mat X_dotdot;             // Demeaned weighted design matrix
  vec v_dotdot;             // Demeaned weighted variable

  ModelResults(size_t n, size_t p)
      : XtX(p, p, fill::none),
        XtY(p, fill::none),
        decomp(p, p, fill::none),
        work(p, fill::none),
        Xt(p, n, fill::none),
        Q(p, 0, fill::none),
        XW(n, 0, fill::none),
        coefficients(p, fill::zeros),
        coef_status(p, fill::ones),
        fitted_values(n, fill::none),
        residuals(n, fill::none),
        weights(n, fill::none),
        hessian(p, p, fill::none),
        // scores(n, p, fill::none),
        success(false),
        // GLM-specific initialization
        eta(n, fill::none),
        mu(n, fill::none),
        deviance(0.0),
        null_deviance(0.0),
        iter(0),
        conv(false),
        residuals_working(n, fill::none),
        residuals_response(n, fill::none),
        has_fe(false),
        // GLM workspace initialization
        W(n, fill::none),
        W_tilde(n, fill::none),
        Z(n, fill::none),
        v_tilde(n, fill::none),
        X_tilde(n, p, fill::none),
        X_dotdot(n, p, fill::none),
        v_dotdot(n, fill::none) {}

  // Template method to copy GLM-specific results to any result structure
  template<typename ResultType>
  void copy_glm_results_to(ResultType &result) const {
    result.coefficients = coefficients;
    result.eta = eta;
    result.fitted_values = mu.n_elem > 0 ? mu : fitted_values;
    result.weights = W_tilde.n_elem > 0 ? W_tilde : weights;
    result.hessian = hessian;
    result.coef_status = coef_status;
    result.conv = conv;
    result.iter = iter;
    result.deviance = deviance;
    result.null_deviance = null_deviance;
    
    // PPML-specific fields (only copy if they exist in both structures)
    if (residuals_working.n_elem > 0) {
      result.residuals_working = residuals_working;
    }
    if (residuals_response.n_elem > 0) {
      result.residuals_response = residuals_response;
    }
    // if (scores.n_elem > 0) {
    //   result.scores = scores;
    // }
    if (has_fe && fixed_effects.n_elem > 0) {
      result.fixed_effects = fixed_effects;
      result.has_fe = true;
    }
  }
};

// Typedef for consistency with usage in other files
using beta_results = ModelResults;

// Solve for regression coefficients using QR decomposition (handles
// collinearity)
inline void get_beta_qr(mat &MX, const vec &MNU, const vec &w, ModelResults &ws,
                        const uword p, bool use_weights, double collin_tol) {
  if (use_weights) {
    if (ws.XW.n_rows != MX.n_rows || ws.XW.n_cols != MX.n_cols) {
      ws.XW.set_size(MX.n_rows, MX.n_cols);
    }

    ws.XW = MX.each_col() % sqrt(w);
    qr_econ(ws.Q, ws.decomp, ws.XW);
  } else {
    qr_econ(ws.Q, ws.decomp, MX);
  }

  ws.work = ws.Q.t() * MNU;

  const vec diag_abs = abs(ws.decomp.diag());
  const double max_diag = diag_abs.max();
  const double tol = collin_tol * max_diag;
  const uvec indep = find(diag_abs > tol);

  ws.coefficients.fill(datum::nan);
  ws.coef_status.zeros();
  ws.coef_status(indep).ones();

  if (indep.n_elem == p) {
    ws.coefficients = solve(trimatu(ws.decomp), ws.work, solve_opts::fast);
  } else if (!indep.is_empty()) {
    const mat Rr = ws.decomp.submat(indep, indep);
    const vec Yr = ws.work.elem(indep);
    const vec br = solve(trimatu(Rr), Yr, solve_opts::fast);
    ws.coefficients(indep) = br;
  }
}

// Use Cholesky if possible, otherwise use QR
inline void get_beta(mat &MX, const vec &MNU, const vec &y_orig, const vec &w,
                     const uword n, const uword p, ModelResults &ws,
                     bool use_weights, double collin_tol,
                     bool has_fixed_effects = false) {
  ws.coefficients.set_size(p);
  ws.coefficients.fill(datum::nan);
  ws.coef_status.zeros(p);
  ws.success = false;

  const bool direct_qr = (p > 0.9 * n);

  if (direct_qr) {
    get_beta_qr(MX, MNU, w, ws, p, use_weights, collin_tol);
  } else {
    if (use_weights) {
      const vec sqrt_w = sqrt(w);
      ws.XW = MX.each_col() % sqrt_w;
      ws.XtX = ws.XW.t() * ws.XW;
      ws.XtY = MX.t() * (w % MNU);
    } else {
      ws.XtX = MX.t() * MX;
      ws.XtY = MX.t() * MNU;
    }

    const bool chol_ok = chol(ws.decomp, ws.XtX, "lower");

    if (chol_ok) {
      const vec d = abs(ws.decomp.diag());
      const double mind = d.min();
      const double avgd = mean(d);

      if (mind > 1e-12 * avgd) {
        ws.work = solve(trimatl(ws.decomp), ws.XtY, solve_opts::fast);
        ws.coefficients =
            solve(trimatu(ws.decomp.t()), ws.work, solve_opts::fast);
        ws.coef_status.ones();
      } else {
        get_beta_qr(MX, MNU, w, ws, p, use_weights, collin_tol);
      }
    } else {
      get_beta_qr(MX, MNU, w, ws, p, use_weights, collin_tol);
    }
  }

  // 1. Fitted values
  if (has_fixed_effects) {
    vec prediction_demeaned = MX * ws.coefficients;
    ws.fitted_values = y_orig - (MNU - prediction_demeaned);
  } else {
    ws.fitted_values = MX * ws.coefficients;
  }

  // 2. Residuals
  ws.residuals = y_orig - ws.fitted_values;

  if (use_weights) {
    ws.residuals = ws.residuals / sqrt(w);
  }

  // 3. Weights
  ws.weights = w;

  // 4. Hessian
  if (use_weights) {
    mat wX = MX.each_col() % w;
    ws.hessian = MX.t() * wX;
  } else {
    ws.hessian = MX.t() * MX;
  }

  // 5. Scores
  // if (use_weights) {
  //   ws.scores = (MX.each_col() % sqrt(w)).each_col() % ws.residuals;
  // } else {
  //   ws.scores = MX.each_col() % ws.residuals;
  // }

  ws.success = true;
}

// Enhanced get_beta for GLM that populates additional GLM-specific fields
inline void get_beta_glm(const mat &MX, const vec &MNU, const vec &y_orig, const vec &w,
                         const uword n, const uword p, ModelResults &ws,
                         bool use_weights, double collin_tol,
                         bool has_fixed_effects = false,
                         const vec &eta_in = vec(), const vec &mu_in = vec()) {
  // Create a non-const copy for get_beta (since it may modify matrices internally)
  mat MX_copy = MX;
  
  // First call regular get_beta to get coefficients and basic results
  get_beta(MX_copy, MNU, y_orig, w, n, p, ws, use_weights, collin_tol, has_fixed_effects);
  
  if (!ws.success) return;
  
  // Store GLM-specific fields if provided
  if (eta_in.n_elem == n) {
    ws.eta = eta_in;
  }
  
  if (mu_in.n_elem == n) {
    ws.mu = mu_in;
  }
  
  // For GLM, compute working and response residuals
  if (use_weights) {
    // Working residuals (for IRLS)
    ws.residuals_working = MNU - MX_copy * ws.coefficients;
    
    // Response residuals
    ws.residuals_response = y_orig - (mu_in.n_elem == n ? mu_in : ws.fitted_values);
    
    // Compute scores matrix for GLM (X * residuals)
    mat WX = MX_copy.each_col() % sqrt(w);
    // ws.scores = WX.each_col() % ws.residuals_working;
  }
}

#endif  // CAPYBARA_BETA
