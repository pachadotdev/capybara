#include <armadillo.hpp>
#include <cmath>
#include <cpp11.hpp>
#include <cpp11armadillo.hpp>
#include <limits>
#include <regex>
#include <unordered_map>

// using namespace arma;
using arma::field;
using arma::mat;
using arma::uvec;
using arma::uword;
using arma::vec;

// using namespace cpp11;
using cpp11::doubles;
using cpp11::doubles_matrix;
using cpp11::integers;
using cpp11::list;

// used across the scripts

void center_variables_(mat &V, const vec &w, const list &klist,
                       const double &tol, const int &max_iter,
                       const int &iter_interrupt, const int &iter_ssr);

vec solve_beta_(mat &MX, const mat &MNU, const vec &w);

vec solve_eta_(const mat &MX, const mat &MNU, const vec &nu, const vec &beta);

mat crossprod_(const mat &X, const vec &w);

std::string tidy_family_(const std::string &family);

enum FamilyType {
  GAUSSIAN,
  POISSON,
  BINOMIAL,
  GAMMA,
  INV_GAUSSIAN,
  NEG_BIN,
  UNKNOWN
};

FamilyType get_family_type(const std::string &fam);

vec link_inv_(const vec &eta, const FamilyType family_type);

double dev_resids_(const vec &y, const vec &mu, const double &theta,
                   const vec &wt, const FamilyType family_type);

vec mu_eta_(const vec &eta, const FamilyType family_type);

vec variance_(const vec &mu, const double &theta, const FamilyType family_type);

bool valid_eta_mu_(const vec &eta, const vec &mu, const FamilyType family_type);

////////////////////////////////////////////////////////////////////////////////
// SECTION: LINEAR ALGEBRA
////////////////////////////////////////////////////////////////////////////////

// Check if the rank of R is less than p
// Demmel Ch. 3: If m >> n, QR and SVD have similar cost. Otherwise, QR is a bit
// cheaper.
// Armadillo's rank() uses SVD, here we count non-zero pivots with an econ-QR
[[cpp11::register]] int check_linear_dependence_qr_(const doubles &y,
                                                    const doubles_matrix<> &x,
                                                    const int &p) {
  mat X = as_mat(x);
  X = join_rows(X, as_mat(y));

  mat Q, R;
  if (!qr_econ(Q, R, X)) {
    stop("QR decomposition failed");
  }

  double tol_qr = std::numeric_limits<double>::epsilon() *
                  std::max(X.n_rows, X.n_cols) * norm(R, "inf");
  int r = accu(arma::abs(diagvec(R)) > tol_qr);

  return (r < p) ? 1 : 0;
}

mat crossprod_(const mat &X, const vec &w) {
  if (w.n_elem == 1) {
    return X.t() * X;
  } else {
    return X.t() * diagmat(w) * X;
  }
}

// Cholesky decomposition
vec solve_beta_(mat &MX, const mat &MNU, const vec &w) {
  mat MXW = MX.t() * diagmat(w);
  mat XtX = MXW * MX;
  vec XtY = MXW * MNU;

  // XtX = L * L.t()
  mat L;
  if (!chol(L, XtX, "lower")) {
    stop("Cholesky decomposition failed.");
  }

  // Solve L * z = Xty
  vec z = solve(trimatl(L), XtY, solve_opts::fast);

  // Solve Lt * beta = z
  return solve(trimatu(L.t()), z, solve_opts::fast);
}

////////////////////////////////////////////////////////////////////////////////
// SECTION: CENTER VARIABLES
////////////////////////////////////////////////////////////////////////////////

// Method of alternating projections (Halperin)
void center_variables_(mat &V, const vec &w, const list &klist,
                       const double &tol, const int &max_iter,
                       const int &iter_interrupt, const int &iter_ssr) {
  // Auxiliary variables (fixed)
  const size_t I = static_cast<size_t>(max_iter), N = V.n_rows, P = V.n_cols,
               K = klist.size(), iint0 = static_cast<size_t>(iter_interrupt),
               isr0 = static_cast<size_t>(iter_ssr);
  const double inv_sw = 1.0 / accu(w);

  // Auxiliary variables (storage)
  size_t iter, iint, isr, j, k, l, p, J, L, group_size, max_groups = 0;
  double coef, ratio, ssr, ssq, ratio0, ssr0, dx_norm, x0_norm, mean_val;
  vec x(N, fill::none), x0(N, fill::none), Gx(N, fill::none),
      G2x(N, fill::none), deltaG(N, fill::none), delta2(N, fill::none),
      diff(N, fill::none), group_means;
  field<field<uvec>> group_indices(K);
  field<vec> group_inv_w(K);

  // Precompute groups into fields
  for (k = 0; k < K; ++k) {
    const list &jlist = klist[k];
    J = jlist.size();
    max_groups = std::max(max_groups, J);
    field<uvec> idxs(J);
    vec invs(J);
    for (j = 0; j < J; ++j) {
      idxs(j) = as_uvec(jlist[j]);
      invs(j) = 1.0 / accu(w(idxs(j)));
    }
    group_indices(k) = std::move(idxs);
    group_inv_w(k) = std::move(invs);
  }

  // Pre-allocate group means vector with maximum size needed
  group_means.set_size(max_groups);

  // Projection step
  auto project = [&](vec &v) {
    // Create a map of group sizes to indices
    std::unordered_map<size_t, std::vector<std::pair<size_t, size_t>>>
        size_groups;

    for (k = 0; k < K; ++k) {
      const auto &idxs = group_indices(k);
      L = idxs.n_elem;
      if (L == 0)
        continue;
      for (l = 0; l < L; ++l) {
        group_size = idxs(l).n_elem;
        size_groups[group_size].push_back({k, l});
      }
    }

    // Process groups of the same size together
    for (const auto &[size, groups] : size_groups) {
      // Batch process all groups of this size
      for (const auto &[k, l] : groups) {
        const uvec &coords = group_indices(k)(l);
        mean_val = as_scalar(w(coords).t() * v(coords)) * group_inv_w(k)(l);
        v(coords) -= mean_val;
      }
    }
  };

  // Column-wise centering with acceleration and SSR checks
  for (p = 0; p < P; ++p) {
    x = V.col(p);
    ratio0 = std::numeric_limits<double>::infinity();
    ssr0 = std::numeric_limits<double>::infinity();
    iint = iint0;
    isr = isr0;

    for (iter = 0; iter < I; ++iter) {
      if (iter == iint) {
        check_user_interrupt();
        iint += iint0;
      }

      x0 = x;
      project(x);

      // 1) convergence via L2 norm
      dx_norm = norm(x - x0, 2);
      x0_norm = norm(x0, 2);
      ratio = dx_norm / (1.0 + x0_norm);

      if (ratio < tol)
        break;

      // 2) Irons-Tuck acceleration every 5 iters
      if (iter >= 5 && (iter % 5) == 0) {
        Gx = x;
        project(Gx);
        G2x = Gx;
        project(G2x);

        deltaG = G2x - x;
        delta2 = G2x - 2.0 * x + x0;
        ssq = dot(delta2, delta2);

        if (ssq > 1e-10) {
          coef = dot(deltaG, delta2) / ssq;
          if (coef > 0.0 && coef < 2.0) {
            x = G2x - coef * deltaG;
          } else {
            x = G2x;
          }
        }
      }

      // 3) SSR-based early exit
      if (iter == isr && iter > 0) {
        check_user_interrupt();
        isr += isr0;
        ssr = dot(x % x, w) * inv_sw;
        if (std::fabs(ssr - ssr0) / (1.0 + std::fabs(ssr0)) < tol)
          break;
        ssr0 = ssr;
      }

      // 4) heuristic early exit
      if (iter > 3 && (ratio0 / ratio) < 1.1 && ratio < tol * 20)
        break;
      ratio0 = ratio;
    }

    V.col(p) = x;
  }
}

[[cpp11::register]] doubles_matrix<>
center_variables_r_(const doubles_matrix<> &V_r, const doubles &w_r,
                    const list &klist, const double &tol, const int &max_iter,
                    const int &iter_interrupt, const int &iter_ssr) {
  mat V = as_mat(V_r);
  center_variables_(V, as_col(w_r), klist, tol, max_iter, iter_interrupt,
                    iter_ssr);
  return as_doubles_matrix(V);
}

////////////////////////////////////////////////////////////////////////////////
// SECTION: LINEAR MODEL
////////////////////////////////////////////////////////////////////////////////

[[cpp11::register]] list felm_fit_(const doubles &y_r,
                                   const doubles_matrix<> &x_r,
                                   const doubles &wt_r, const list &control,
                                   const list &k_list) {
  // Type conversion

  mat X = as_Mat(x_r);
  const vec y = as_Col(y_r);
  const vec w = as_Col(wt_r);

  // Auxiliary variables (fixed)

  const double center_tol = as_cpp<double>(control["center_tol"]);
  const size_t iter_center_max = as_cpp<size_t>(control["iter_center_max"]),
               iter_interrupt = as_cpp<size_t>(control["iter_interrupt"]),
               iter_ssr = as_cpp<size_t>(control["iter_ssr"]);

  // Auxiliary variables (storage)

  mat H(X.n_cols, X.n_cols, fill::none);
  vec MNU(y.n_elem, fill::none), beta(X.n_cols, fill::none),
      fitted(y.n_elem, fill::none);

  // Center variables

  const bool has_fixed_effects = k_list.size() > 0;

  if (has_fixed_effects) {
    // Initial response + centering for fixed effects
    MNU = y;
    center_variables_(MNU, w, k_list, center_tol, iter_center_max,
                      iter_interrupt, iter_ssr);
    center_variables_(X, w, k_list, center_tol, iter_center_max, iter_interrupt,
                      iter_ssr);
  } else {
    // No fixed effects
    MNU = vec(y.n_elem, fill::zeros);
  }

  // Solve the normal equations

  beta = solve_beta_(X, MNU, w);

  // Fitted values

  if (has_fixed_effects) {
    fitted = y - MNU + X * beta;
  } else {
    fitted = X * beta;
  }

  // Recompute Hessian

  H = crossprod_(std::move(X), w);

  // Generate result list

  return writable::list({"coefficients"_nm = as_doubles(beta),
                         "fitted.values"_nm = as_doubles(fitted),
                         "weights"_nm = as_doubles(w),
                         "hessian"_nm = as_doubles_matrix(H)});
}

////////////////////////////////////////////////////////////////////////////////
// SECTION: GENERALIZED LINEAR MODEL
////////////////////////////////////////////////////////////////////////////////

std::string tidy_family_(const std::string &family) {
  // tidy family param
  std::string fam = family;

  // 1. put all in lowercase
  std::transform(fam.begin(), fam.end(), fam.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  // 2. remove numbers
  fam.erase(std::remove_if(fam.begin(), fam.end(), ::isdigit), fam.end());

  // 3. remove parentheses and everything inside
  size_t pos = fam.find("(");
  if (pos != std::string::npos) {
    fam.erase(pos, fam.size());
  }

  // 4. replace spaces and dots
  std::replace(fam.begin(), fam.end(), ' ', '_');
  std::replace(fam.begin(), fam.end(), '.', '_');

  // 5. trim
  fam.erase(std::remove_if(fam.begin(), fam.end(), ::isspace), fam.end());

  return fam;
}

FamilyType get_family_type(const std::string &fam) {
  static const std::unordered_map<std::string, FamilyType> family_map = {
      {"gaussian", GAUSSIAN},
      {"poisson", POISSON},
      {"binomial", BINOMIAL},
      {"gamma", GAMMA},
      {"inverse_gaussian", INV_GAUSSIAN},
      {"negative_binomial", NEG_BIN}};

  auto it = family_map.find(fam);
  return (it != family_map.end()) ? it->second : UNKNOWN;
}

vec link_inv_gaussian_(const vec &eta) { return eta; }

vec link_inv_poisson_(const vec &eta) { return exp(eta); }

vec link_inv_logit_(const vec &eta) { return 1.0 / (1.0 + exp(-eta)); }

vec link_inv_gamma_(const vec &eta) { return 1 / eta; }

vec link_inv_invgaussian_(const vec &eta) { return 1 / arma::sqrt(eta); }

vec link_inv_negbin_(const vec &eta) { return exp(eta); }

double dev_resids_gaussian_(const vec &y, const vec &mu, const vec &wt) {
  return dot(wt, square(y - mu));
}

double dev_resids_poisson_(const vec &y, const vec &mu, const vec &wt) {
  vec r = mu % wt;

  uvec p = find(y > 0);
  r(p) = wt(p) % (y(p) % log(y(p) / mu(p)) - (y(p) - mu(p)));

  return 2 * accu(r);
}

// Adapted from binomial_dev_resids()
// in R base it can be found in src/library/stats/src/family.c
// unfortunately the functions that work with a SEXP won't work with a Col<>
double dev_resids_logit_(const vec &y, const vec &mu, const vec &wt) {
  vec r(y.n_elem, fill::none);

  uvec p = find(y == 1);
  uvec q = find(y == 0);
  vec y_p = y(p), y_q = y(q);

  r(p) = y_p % log(y_p / mu(p));
  r(q) = (1 - y_q) % log((1 - y_q) / (1 - mu(q)));

  return 2 * dot(wt, r);
}

double dev_resids_gamma_(const vec &y, const vec &mu, const vec &wt) {
  vec r = y / mu;

  uvec p = find(y == 0);
  r(p).fill(1.0);
  r = wt % (log(r) - (y - mu) / mu);

  return -2 * accu(r);
}

double dev_resids_invgaussian_(const vec &y, const vec &mu, const vec &wt) {
  return dot(wt, square(y - mu) / (y % square(mu)));
}

double dev_resids_negbin_(const vec &y, const vec &mu, const double &theta,
                          const vec &wt) {
  vec r = y;

  uvec p = find(y < 1);
  r(p).fill(1.0);
  r = wt % (y % log(r / mu) - (y + theta) % log((y + theta) / (mu + theta)));

  return 2 * accu(r);
}

vec variance_gaussian_(const vec &mu) { return ones<vec>(mu.n_elem); }

vec link_inv_(const vec &eta, const FamilyType family_type) {
  vec result(eta.n_elem);

  switch (family_type) {
  case GAUSSIAN:
    result = link_inv_gaussian_(eta);
    break;
  case POISSON:
    result = link_inv_poisson_(eta);
    break;
  case BINOMIAL:
    result = link_inv_logit_(eta);
    break;
  case GAMMA:
    result = link_inv_gamma_(eta);
    break;
  case INV_GAUSSIAN:
    result = link_inv_invgaussian_(eta);
    break;
  case NEG_BIN:
    result = link_inv_negbin_(eta);
    break;
  default:
    stop("Unknown family");
  }

  return result;
}

double dev_resids_(const vec &y, const vec &mu, const double &theta,
                   const vec &wt, const FamilyType family_type) {
  switch (family_type) {
  case GAUSSIAN:
    return dev_resids_gaussian_(y, mu, wt);
  case POISSON:
    return dev_resids_poisson_(y, mu, wt);
  case BINOMIAL:
    return dev_resids_logit_(y, mu, wt);
  case GAMMA:
    return dev_resids_gamma_(y, mu, wt);
  case INV_GAUSSIAN:
    return dev_resids_invgaussian_(y, mu, wt);
  case NEG_BIN:
    return dev_resids_negbin_(y, mu, theta, wt);
  default:
    stop("Unknown family");
  }
}

bool valid_eta_mu_(const vec &eta, const vec &mu,
                   const FamilyType family_type) {
  switch (family_type) {
  case GAUSSIAN:
    return true;
  case POISSON:
  case NEG_BIN:
    return is_finite(mu) && all(mu > 0);
  case BINOMIAL:
    return is_finite(mu) && all(mu > 0 && mu < 1);
  case GAMMA:
    return is_finite(eta) && all(eta != 0.0) && is_finite(mu) && all(mu > 0.0);
  case INV_GAUSSIAN:
    return is_finite(eta) && all(eta > 0.0);
  default:
    stop("Unknown family");
  }
}

// mu_eta = d link_inv / d eta = d mu / d eta

vec mu_eta_(const vec &eta, const FamilyType family_type) {
  vec result(eta.n_elem);

  switch (family_type) {
  case GAUSSIAN:
    result.ones();
    break;
  case POISSON:
  case NEG_BIN:
    result = arma::exp(eta);
    break;
  case BINOMIAL: {
    vec exp_eta = arma::exp(eta);
    result = exp_eta / arma::square(1 + exp_eta);
    break;
  }
  case GAMMA:
    result = -1 / arma::square(eta);
    break;
  case INV_GAUSSIAN:
    result = -1 / (2 * arma::pow(eta, 1.5));
    break;
  default:
    stop("Unknown family");
  }

  return result;
}

vec variance_(const vec &mu, const double &theta,
              const FamilyType family_type) {
  switch (family_type) {
  case GAUSSIAN:
    return ones<vec>(mu.n_elem);
  case POISSON:
    return mu;
  case BINOMIAL:
    return mu % (1 - mu);
  case GAMMA:
    return square(mu);
  case INV_GAUSSIAN:
    return pow(mu, 3.0);
  case NEG_BIN:
    return mu + square(mu) / theta;
  default:
    stop("Unknown family");
  }
}

[[cpp11::register]] list feglm_fit_(const doubles &beta_r, const doubles &eta_r,
                                    const doubles &y_r,
                                    const doubles_matrix<> &x_r,
                                    const doubles &wt_r, const double &theta,
                                    const std::string &family,
                                    const list &control, const list &k_list) {
  // Type conversion
  mat MX = as_Mat(x_r);
  vec beta = as_Col(beta_r);
  vec eta = as_Col(eta_r);
  const vec y = as_Col(y_r);
  vec MNU = vec(y.n_elem, fill::zeros);
  const vec wt = as_Col(wt_r);

  // Auxiliary variables (fixed)

  const std::string fam = tidy_family_(family);
  const FamilyType family_type = get_family_type(fam);
  const double center_tol = as_cpp<double>(control["center_tol"]),
               dev_tol = as_cpp<double>(control["dev_tol"]);
  const bool keep_mx = as_cpp<bool>(control["keep_mx"]);
  const size_t iter_max = as_cpp<size_t>(control["iter_max"]),
               iter_center_max = as_cpp<size_t>(control["iter_center_max"]),
               iter_inner_max = as_cpp<size_t>(control["iter_inner_max"]),
               iter_interrupt = as_cpp<size_t>(control["iter_interrupt"]),
               iter_ssr = as_cpp<size_t>(control["iter_ssr"]), n = y.n_elem,
               p = MX.n_cols, k = beta.n_elem;

  // Auxiliary variables (storage)

  size_t iter, iter_inner;
  vec mu = link_inv_(eta, family_type),
      ymean = mean(y) * vec(y.n_elem, fill::ones), mu_eta(n, fill::none),
      w(n, fill::none), nu(n, fill::none), beta_upd(k, fill::none),
      eta_upd(n, fill::none), eta_old(n, fill::none), beta_old(k, fill::none),
      nu_old = vec(n, fill::zeros);
  mat H(p, p, fill::none);
  double dev = dev_resids_(y, mu, theta, wt, family_type),
         null_dev = dev_resids_(y, ymean, theta, wt, family_type), dev_old,
         dev_ratio, dev_ratio_inner, rho;
  bool dev_crit, val_crit, imp_crit, conv = false;

  // Maximize the log-likelihood
  for (iter = 0; iter < iter_max; ++iter) {
    rho = 1.0;
    eta_old = eta;
    beta_old = beta;
    dev_old = dev;

    // Compute weights and dependent variable
    mu_eta = mu_eta_(eta, family_type);
    w = (wt % square(mu_eta)) / variance_(mu, theta, family_type);
    nu = (y - mu) / mu_eta;

    // Center variables

    MNU += (nu - nu_old);
    nu_old = nu;

    center_variables_(MNU, w, k_list, center_tol, iter_center_max,
                      iter_interrupt, iter_ssr);
    center_variables_(MX, w, k_list, center_tol, iter_center_max,
                      iter_interrupt, iter_ssr);

    // Compute update step and update eta

    // Step-halving with three checks:
    // 1. finite deviance
    // 2. valid eta and mu
    // 3. improvement as in glm2

    beta_upd = solve_beta_(MX, MNU, w);
    eta_upd = MX * beta_upd + nu - MNU;

    for (iter_inner = 0; iter_inner < iter_inner_max; ++iter_inner) {
      eta = eta_old;
      eta += rho * eta_upd;
      beta = beta_old;
      beta += rho * beta_upd;
      mu = link_inv_(eta, family_type);
      dev = dev_resids_(y, mu, theta, wt, family_type);
      dev_ratio_inner = (dev - dev_old) / (0.1 + fabs(dev));

      dev_crit = is_finite(dev);
      val_crit = valid_eta_mu_(eta, mu, family_type);
      imp_crit = (dev_ratio_inner <= -dev_tol);

      if (dev_crit && val_crit && imp_crit) {
        break;
      }

      rho *= 0.5;
    }

    // Check if step-halving failed (deviance and invalid eta or mu)

    if (!dev_crit || !val_crit) {
      stop("Inner loop failed; cannot correct step size.");
    }

    // If step halving does not improve the deviance

    if (!imp_crit) {
      eta = eta_old;
      beta = beta_old;
      dev = dev_old;
      mu = link_inv_(eta, family_type);
    }

    // Check convergence

    dev_ratio = fabs(dev - dev_old) / (0.1 + fabs(dev));

    if (dev_ratio < dev_tol) {
      conv = true;
      break;
    }
  }

  // Information if convergence failed
  if (!conv) {
    stop("Algorithm did not converge.");
  }

  // Compute Hessian

  H = crossprod_(std::move(MX), w);

  // Generate result list

  writable::list out = writable::list({
      "coefficients"_nm = as_doubles(beta),
      "eta"_nm = as_doubles(eta),
      "weights"_nm = as_doubles(wt),
      "hessian"_nm = as_doubles_matrix(H),
      "deviance"_nm = writable::doubles({dev}),
      "null_deviance"_nm = writable::doubles({null_dev}),
      "conv"_nm = writable::logicals({conv}),
      "iter"_nm = writable::integers({static_cast<int>(iter + 1)}),
  });

  if (keep_mx) {
    mat x_cpp = as_Mat(x_r);
    center_variables_(x_cpp, w, k_list, center_tol, iter_center_max,
                      iter_interrupt, iter_ssr);
    out.push_back({"MX"_nm = as_doubles_matrix(x_cpp)});
  }

  return out;
}

////////////////////////////////////////////////////////////////////////////////
// SECTION: GLM OFFSET
////////////////////////////////////////////////////////////////////////////////

[[cpp11::register]] doubles
feglm_offset_fit_(const doubles &eta_r, const doubles &y_r,
                  const doubles &offset_r, const doubles &wt_r,
                  const std::string &family, const list &control,
                  const list &k_list) {
  // Type conversion

  vec eta = as_Col(eta_r);
  vec y = as_Col(y_r);
  const vec offset = as_Col(offset_r);
  vec Myadj = vec(y.n_elem, fill::zeros);
  const vec wt = as_Col(wt_r);

  // Auxiliary variables (fixed)

  const std::string fam = tidy_family_(family);
  const FamilyType family_type = get_family_type(fam);
  const double center_tol = as_cpp<double>(control["center_tol"]),
               dev_tol = as_cpp<double>(control["dev_tol"]);
  const size_t iter_max = as_cpp<int>(control["iter_max"]),
               iter_center_max = as_cpp<size_t>(control["iter_center_max"]),
               iter_inner_max = as_cpp<size_t>(control["iter_inner_max"]),
               iter_interrupt = as_cpp<size_t>(control["iter_interrupt"]),
               iter_ssr = as_cpp<size_t>(control["iter_ssr"]), n = y.n_elem;

  // Auxiliary variables (storage)

  size_t iter, iter_inner;
  vec mu = link_inv_(eta, family_type), mu_eta(n, fill::none),
      yadj(n, fill::none), w(n, fill::none), eta_upd(n, fill::none),
      eta_old(n, fill::none);
  double dev = dev_resids_(y, mu, 0.0, wt, family_type), dev_old, dev_ratio,
         dev_ratio_inner, rho;
  bool dev_crit, val_crit, imp_crit;

  // Maximize the log-likelihood

  for (iter = 0; iter < iter_max; ++iter) {
    rho = 1.0;
    eta_old = eta, dev_old = dev;

    // Compute weights and dependent variable

    mu_eta = mu_eta_(eta, family_type);
    w = (wt % square(mu_eta)) / variance_(mu, 0.0, family_type);
    yadj = (y - mu) / mu_eta + eta - offset;

    // Center variables

    Myadj += yadj;
    center_variables_(Myadj, w, k_list, center_tol, iter_center_max,
                      iter_interrupt, iter_ssr);

    // Compute update step and update eta

    // Step-halving with three checks:
    // 1. finite deviance
    // 2. valid eta and mu
    // 3. improvement as in glm2

    eta_upd = yadj - Myadj + offset - eta;

    for (iter_inner = 0; iter_inner < iter_inner_max; ++iter_inner) {
      eta = eta_old + (rho * eta_upd);
      mu = link_inv_(eta, family_type);
      dev = dev_resids_(y, mu, 0.0, wt, family_type);
      dev_ratio_inner = (dev - dev_old) / (0.1 + fabs(dev_old));

      dev_crit = is_finite(dev);
      val_crit = valid_eta_mu_(eta, mu, family_type);
      imp_crit = (dev_ratio_inner <= -dev_tol);

      if (dev_crit == true && val_crit == true && imp_crit == true) {
        break;
      }

      rho *= 0.5;
    }

    // Check if step-halving failed (deviance and invalid eta or mu)

    if (dev_crit == false || val_crit == false) {
      stop("Inner loop failed; cannot correct step size.");
    }

    // Check convergence

    dev_ratio = fabs(dev - dev_old) / (0.1 + fabs(dev));

    if (dev_ratio < dev_tol) {
      break;
    }

    // Update starting guesses for acceleration

    Myadj = Myadj - yadj;
  }

  return as_doubles(eta);
}

////////////////////////////////////////////////////////////////////////////////
// SECTION: RECOVER FIXED EFFECTS
////////////////////////////////////////////////////////////////////////////////

[[cpp11::register]] list get_alpha_(const doubles_matrix<> &p_r,
                                    const list &klist, const list &control) {
  // Types conversion
  const vec p = as_Mat(p_r);

  // Auxiliary variables (fixed)
  const double tol = as_cpp<double>(control["center_tol"]);
  const size_t K = klist.size(), iter_max = as_cpp<int>(control["iter_max"]),
               interrupt_iter0 = as_cpp<size_t>(control["iter_interrupt"]);

  // Auxiliary variables (storage)
  size_t j, k, l, iter, J, J1, J2, interrupt_iter = interrupt_iter0;
  double num, denom, ratio;
  vec y(p.n_elem, fill::none), subtract_vec(p.n_elem, fill::none);

  // Pre-compute list sizes
  field<int> list_sizes(K);
  field<field<uvec>> group_indices(K);

  for (k = 0; k < K; ++k) {
    const list &jlist = as_cpp<list>(klist[k]);
    J = jlist.size();
    list_sizes(k) = J;
    group_indices(k).set_size(J);
    for (j = 0; j < J; ++j) {
      group_indices(k)(j) = as_uvec(jlist[j]);
    }
  }

  // Generate starting guess
  field<vec> Alpha(K), Alpha0(K);
  for (k = 0; k < K; ++k) {
    if (list_sizes(k) > 0) {
      Alpha(k).zeros(list_sizes(k));
      Alpha0(k).zeros(list_sizes(k));
    }
  }

  // Start alternating between normal equations
  for (iter = 0; iter < iter_max; ++iter) {
    if (iter == interrupt_iter) {
      check_user_interrupt();
      interrupt_iter += interrupt_iter0;
    }

    // Store alpha_0 of the previous iteration
    Alpha0 = Alpha;

    for (k = 0; k < K; ++k) {
      if (list_sizes(k) == 0)
        continue;

      // Compute adjusted dependent variable
      y = p;

      for (l = 0; l < K; ++l) {
        if (l == k || list_sizes(l) == 0)
          continue;

        const field<uvec> &indices_l = group_indices(l);
        const vec &alpha_l = Alpha0(l);

        J1 = list_sizes(l);
        for (j = 0; j < J1; ++j) {
          subtract_vec(indices_l(j)).fill(alpha_l(j));
        }
        y -= subtract_vec;
      }

      J2 = list_sizes(k);
      for (j = 0; j < J2; ++j) {
        Alpha(k)(j) = mean(y(group_indices(k)(j)));
      }
    }

    // Compute termination criterion and check convergence
    num = 0.0, denom = 0.0;

    for (k = 0; k < K; ++k) {
      if (list_sizes(k) == 0)
        continue; // Skip empty groups
      const vec &diff = Alpha(k) - Alpha0(k);
      num += dot(diff, diff);
      denom += dot(Alpha0(k), Alpha0(k));
    }

    ratio = std::sqrt(num / denom + 1e-16);
    if (ratio < tol) {
      break;
    }
  }

  // Return alpha
  writable::list Alpha_r(K);
  for (k = 0; k < K; ++k) {
    Alpha_r[k] = as_doubles_matrix(Alpha(k).eval()); // Ensure materialization
  }

  return Alpha_r;
}

////////////////////////////////////////////////////////////////////////////////
// SECTION: COVARIANCE MATRICES
////////////////////////////////////////////////////////////////////////////////

[[cpp11::register]] doubles_matrix<> group_sums_(const doubles_matrix<> &M_r,
                                                 const doubles_matrix<> &w_r,
                                                 const list &jlist) {
  // Types conversion
  const mat M = as_Mat(M_r);
  const mat w = as_Mat(w_r);

  // Auxiliary variables (fixed)
  const size_t J = jlist.size(), P = M.n_cols;

  // Auxiliary variables (storage)
  size_t j;
  uvec indexes;
  mat b(P, 1, fill::zeros), groupSum(P, 1, fill::none);
  double w_sum;

  // Compute sum of weighted group sums
  for (j = 0; j < J; ++j) {
    indexes = as_uvec(jlist[j]);

    if (indexes.n_elem > 0) {
      groupSum = M.rows(indexes).t() * ones(indexes.n_elem);

      w_sum = accu(w(indexes));
      b += groupSum / w_sum;
    }
  }

  return as_doubles_matrix(b);
}

[[cpp11::register]] doubles_matrix<>
group_sums_spectral_(const doubles_matrix<> &M_r, const doubles_matrix<> &v_r,
                     const doubles_matrix<> &w_r, const int K,
                     const list &jlist) {
  // Types conversion
  const mat M = as_Mat(M_r);
  const mat v = as_Mat(v_r);
  const mat w = as_Mat(w_r);

  // Auxiliary variables (fixed)
  const size_t J = jlist.size(), K1 = K, P = M.n_cols;

  // Auxiliary variables (storage)
  size_t j, k, I;
  uvec indexes;
  vec num(P, fill::none), v_shifted;
  mat b(P, 1, fill::zeros);
  double denom;

  // Compute sum of weighted group sums
  for (j = 0; j < J; ++j) {
    indexes = as_uvec(jlist[j]);
    I = indexes.n_elem;

    if (I <= 1)
      continue;

    num.fill(0.0);

    denom = accu(w(indexes));

    v_shifted.zeros(I);
    vec v_indexed = v(indexes);
    for (k = 1; k <= K1 && k < I; ++k) {
      v_shifted.subvec(k, I - 1) += v_indexed.subvec(0, I - k - 1);
    }

    num = M.rows(indexes).t() * (v_shifted * (I / (I - 1.0)));

    b += num / denom;
  }

  return as_doubles_matrix(b);
}

[[cpp11::register]] doubles_matrix<>
group_sums_var_(const doubles_matrix<> &M_r, const list &jlist) {
  // Types conversion
  const mat M = as_Mat(M_r);

  // Auxiliary variables (fixed)
  const int J = jlist.size();
  const int P = M.n_cols;

  // Auxiliary variables (storage)
  int j;
  mat v(P, 1, fill::none), V(P, P, fill::zeros);
  uvec indexes;

  // Compute covariance matrix
  for (j = 0; j < J; ++j) {
    indexes = as_uvec(jlist[j]);

    v = sum(M.rows(indexes), 0).t();

    V += v * v.t();
  }

  return as_doubles_matrix(V);
}

[[cpp11::register]] doubles_matrix<>
group_sums_cov_(const doubles_matrix<> &M_r, const doubles_matrix<> &N_r,
                const list &jlist) {
  // Types conversion
  const mat M = as_Mat(M_r);
  const mat N = as_Mat(N_r);

  // Auxiliary variables (fixed)
  const int J = jlist.size();
  const int P = M.n_cols;

  // Auxiliary variables (storage)
  int j;
  uvec indexes;
  mat V(P, P, fill::zeros);

  // Compute covariance matrix
  for (j = 0; j < J; ++j) {
    indexes = as_uvec(jlist[j]);

    if (indexes.n_elem < 2) {
      continue;
    }

    V += M.rows(indexes).t() * N.rows(indexes);
  }

  return as_doubles_matrix(V);
}
