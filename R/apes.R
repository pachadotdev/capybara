#' srr_stats
#' @srrstats {G1.0} Closely follows methodologies described in Stammann (2018) and other referenced works for binary choice models.
#' @srrstats {G2.1a} Ensures the input object is of the expected class (`bias_corr` or `feglm`).
#' @srrstats {G2.3a} Uses `match.arg()` to validate `panel_structure` and `sampling_fe` inputs against expected values.
#' @srrstats {G2.3b} Uses `tolower()` to handle potential case sensitivity issues.
#' @srrstats {G2.13} Validates that the input data contains no missing values.
#' @srrstats {G2.14a} Issues errors when handling missing data is required but unsupported.
#' @srrstats {G2.14b} Provides clear error messages when the data structure is incompatible with the model requirements.
#' @srrstats {G3.1a} Allows arbitrarily specified covariance methods for flexibility in inference.
#' @srrstats {G5.2a} Produces unique and meaningful error, warning, and message outputs for diagnostics.
#' @srrstats {RE5.0} Considers relationships between input data size and computational efficiency.
#' @srrstats {G5.4a} Includes tests against trivial cases or alternative implementations to ensure algorithm correctness.
#' @noRd
NULL

#' NA_standards
#' @srrstatsNA {G2.14} Missing observations are dropped, otherwise providing imputation methods would bias the estimation (i.e., replacing all missing values with the median).
#' @noRd
NULL

#' @title Compute average partial effects after fitting binary choice models
#'  with a 1,2,3-way error component
#'
#' @description \code{\link{apes}} is a post-estimation routine that can be used
#'  to estimate average partial effects with respect to all covariates in the
#'  model and the corresponding covariance matrix. The estimation of the
#'  covariance is based on a linear approximation (delta method) plus an
#'  optional finite population correction. Note that the command automatically
#'  determines which of the regressors are binary or non-binary.
#'
#'  \strong{Remark:} The routine currently does not allow to compute average
#'  partial effects based on functional forms like interactions and polynomials.
#'
#' @param object an object of class \code{"bias_corr"} or \code{"feglm"};
#'  currently restricted to \code{\link[stats]{binomial}}.
#' @param n_pop unsigned integer indicating a finite population correction for
#'  the estimation of the covariance matrix of the average partial effects
#'  proposed by Cruz-Gonzalez, Fernández-Val, and Weidner (2017). The correction
#'  factor is computed as follows:
#'  \eqn{(n^{\ast} - n) / (n^{\ast} - 1)}{(n_pop - n) / (n_pop - 1)},
#'  where \eqn{n^{\ast}}{n_pop} and \eqn{n}{n} are the sizes of the entire
#'  population and the full sample size. Default is \code{NULL}, which refers to
#'  a factor of zero and a covariance obtained by the delta method.
#' @param panel_structure a string equal to \code{"classic"} or \code{"network"}
#'  which determines the structure of the panel used. \code{"classic"} denotes
#'  panel structures where for example the same cross-sectional units are
#'  observed several times (this includes pseudo panels). \code{"network"}
#'  denotes panel structures where for example bilateral trade flows are
#'  observed for several time periods. Default is \code{"classic"}.
#' @param sampling_fe a string equal to \code{"independence"} or
#'  \code{"unrestricted"} which imposes sampling assumptions about the
#'  unobserved effects. \code{"independence"} imposes that all unobserved
#'  effects are independent sequences. \code{"unrestricted"} does not impose any
#'  sampling assumptions. Note that this option only affects the optional finite
#'  population correction. Default is \code{"independence"}.
#' @param weak_exo logical indicating if some of the regressors are assumed to
#'  be weakly exogenous (e.g. predetermined). If object is of class
#'  \code{"bias_corr"}, the option will be automatically set to \code{TRUE} if
#'  the chosen bandwidth parameter is larger than zero. Note that this option
#'  only affects the estimation of the covariance matrix. Default is
#'  \code{FALSE}, which assumes that all regressors are strictly exogenous.
#'
#' @return The function \code{\link{apes}} returns a named list of class
#'  \code{"apes"}.
#'
#' @references Cruz-Gonzalez, M., I. Fernández-Val, and M. Weidner (2017). "Bias
#'  corrections for probit and logit models with two-way fixed effects". The
#'  Stata Journal, 17(3), 517-545.
#' @references Czarnowske, D. and A. Stammann (2020). "Fixed Effects Binary
#'  Choice Models: Estimation and Inference with Long Panels". ArXiv e-prints.
#' @references Fernández-Val, I. and M. Weidner (2016). "Individual and time
#'  effects in nonlinear panel models with large N, T". Journal of Econometrics,
#'  192(1), 291-312.
#' @references Fernández-Val, I. and M. Weidner (2018). "Fixed effects
#'  estimation of large-t panel data models". Annual Review of Economics, 10,
#'  109-138.
#' @references Hinz, J., A. Stammann, and J. Wanner (2020). "State Dependence
#'  and Unobserved Heterogeneity in the Extensive Margin of Trade". ArXiv
#'  e-prints.
#' @references Neyman, J. and E. L. Scott (1948). "Consistent estimates based on
#'  partially consistent observations". Econometrica, 16(1), 1-32.
#'
#' @seealso \code{\link{bias_corr}}, \code{\link{feglm}}
#'
#' @examples
#' mtcars2 <- mtcars
#' mtcars2$mpg01 <- ifelse(mtcars2$mpg > mean(mtcars2$mpg), 1L, 0L)
#'
#' # Fit 'feglm()'
#' mod <- feglm(mpg01 ~ wt | cyl, mtcars2, family = binomial())
#'
#' # Compute average partial effects
#' mod_ape <- apes(mod)
#' summary(mod_ape)
#'
#' # Apply analytical bias correction
#' mod_bc <- bias_corr(mod)
#' summary(mod_bc)
#'
#' # Compute bias-corrected average partial effects
#' mod_ape_bc <- apes(mod_bc)
#' summary(mod_ape_bc)
#'
#' @export
apes <- function(
    object = NULL,
    n_pop = NULL,
    panel_structure = c("classic", "network"),
    sampling_fe = c("independence", "unrestricted"),
    weak_exo = FALSE) {
  # Check validity of 'object'
  apes_bias_check_object_(object, fun = "apes")

  # Extract prior information if available or check validity of panel_structure
  bias_corr <- inherits(object, "bias_corr")
  if (bias_corr) {
    panel_structure <- object[["panel_structure"]]
    l <- object[["bandwidth"]]
    if (l > 0L) {
      weak_exo <- TRUE
    } else {
      weak_exo <- FALSE
    }
  } else {
    panel_structure <- match.arg(panel_structure)
  }

  # Check validity of 'sampling_fe'
  sampling_fe <- match.arg(sampling_fe)

  # Extract model information
  beta <- object[["coefficients"]]
  control <- object[["control"]]
  data <- object[["data"]]
  family <- object[["family"]]
  formula <- object[["formula"]]
  lvls_k <- object[["lvls_k"]]
  nt <- nrow(data)
  nt_full <- object[["nobs"]][["nobs_full"]]
  k <- length(lvls_k)
  k_vars <- names(lvls_k)
  p <- length(beta)

  # Check if binary choice model
  apes_bias_check_binary_model_(family, fun = "apes")

  # Check if provided object matches requested panel structure
  apes_bias_check_panel_(panel_structure, k)

  # Check validity of 'n_pop'
  # Note: Default option is no adjustment i.e. only delta method covariance
  adj <- apes_set_adj_(n_pop, nt_full)

  # Extract model response, regressor matrix, and weights
  y <- data[[1L]]
  x <- model.matrix(formula, data, rhs = 1L)[, -1L, drop = FALSE]
  nms_sp <- attr(x, "dimnames")[[2L]]
  attr(x, "dimnames") <- NULL
  wt <- object[["weights"]]

  # Determine which of the regressors are binary
  binary <- apply(x, 2L, function(x) all(x %in% c(0.0, 1.0)))

  # Generate auxiliary list of indexes for different sub panels
  k_list <- get_index_list_(k_vars, data)

  # Compute derivatives and weights
  eta <- object[["eta"]]
  mu <- family[["linkinv"]](eta)
  mu_eta <- family[["mu.eta"]](eta)
  v <- wt * (y - mu)
  w <- wt * mu_eta
  z <- wt * partial_mu_eta_(eta, family, 2L)
  if (family[["link"]] != "logit") {
    h <- mu_eta / family[["variance"]](mu)
    v <- h * v
    w <- h * w
    z <- h * z
    rm(h)
  }

  # Center regressor matrix (if required)
  if (control[["keep_mx"]]) {
    mx <- object[["mx"]]
  } else {
    mx <- center_variables_r_(x, w, k_list, control[["center_tol"]], control[["iter_max"]], control[["iter_interrupt"]], control[["iter_ssr"]])
  }

  # Compute average partial effects, derivatives, and Jacobian
  px <- x - mx
  delta <- matrix(NA_real_, nt, p)
  delta1 <- matrix(NA_real_, nt, p)
  j <- matrix(NA_real_, p, p)
  if (any(!binary)) {
    delta[, !binary] <- mu_eta
    delta1[, !binary] <- partial_mu_eta_(eta, family, 2L)
  }
  for (i in seq.int(p)) {
    if (binary[[i]]) {
      eta0 <- eta - x[, i] * beta[[i]]
      eta1 <- eta0 + beta[[i]]
      f1 <- family[["mu.eta"]](eta1)
      delta[, i] <- (family[["linkinv"]](eta1) - family[["linkinv"]](eta0))
      delta1[, i] <- f1 - family[["mu.eta"]](eta0)
      j[, i] <- -colSums(px * delta1[, i]) / nt_full
      j[i, i] <- sum(f1) / nt_full + j[i, i]
      j[-i, i] <- colSums(x[, -i, drop = FALSE] * delta1[, i]) /
        nt_full + j[-i, i]
      rm(eta0, f1)
    } else {
      delta[, i] <- beta[[i]] * delta[, i]
      delta1[, i] <- beta[[i]] * delta1[, i]
      j[, i] <- colSums(mx * delta1[, i]) / nt_full
      j[i, i] <- sum(mu_eta) / nt_full + j[i, i]
    }
  }
  delta_aux <- colSums(delta) / nt_full
  delta <- t(t(delta) - delta_aux) / nt_full
  rm(mu, mu_eta, px)

  # Compute projection and residual projection of \psi
  psi <- -delta1 / w
  mpsi <- center_variables_r_(psi, w, k_list, control[["center_tol"]], control[["iter_max"]], control[["iter_interrupt"]], control[["iter_ssr"]])
  ppsi <- psi - mpsi
  rm(delta1, psi)

  # Compute analytical bias correction of average partial effects
  if (bias_corr) {
    b <- apes_bias_correction_(
      eta, family, x, beta, binary, nt, p, ppsi, z,
      w, k_list, panel_structure, l, k, mpsi, v
    )
    delta_aux <- delta_aux - b
  }
  rm(eta, w, z, mpsi)

  # Compute covariance matrix
  gamma <- gamma_(mx, object[["hessian"]], j, ppsi, v, nt_full)
  v <- crossprod(gamma)

  v <- apes_adjust_covariance_(
    v, delta, gamma, k_list, adj, sampling_fe,
    weak_exo, panel_structure
  )

  # Add names
  names(delta_aux) <- nms_sp
  dimnames(v) <- list(nms_sp, nms_sp)

  # Generate result list
  reslist <- list(
    delta           = delta_aux,
    vcov            = v,
    panel_structure = panel_structure,
    sampling_fe     = sampling_fe,
    weak_exo        = weak_exo
  )

  # Update result list
  if (bias_corr) {
    names(b) <- nms_sp
    reslist[["bias.term"]] <- b
    reslist[["bandwidth"]] <- l
  }

  # Return result list
  structure(reslist, class = "apes")
}

#' srr_stats
#' @srrstats {G2.0} Implements assertions to ensure valid scaling relationships between population size and sample size.
#' @srrstats {G2.0a} The main function explains that the inputs are unidimensional or the function gives an error.
#' @srrstats {G5.2a} Issues clear warnings for invalid population adjustments or mismatched sizes.
#' @noRd
NULL

apes_set_adj_ <- function(n_pop, nt_full) {
  if (!is.null(n_pop)) {
    n_pop <- as.integer(n_pop)
    if (n_pop < nt_full) {
      warning(
        paste(
          "Size of the entire population is lower than the full sample size.",
          "Correction factor set to zero."
        ),
        call. = FALSE
      )
      adj <- 0.0
    } else {
      adj <- (n_pop - nt_full) / (n_pop - 1L)
    }
  } else {
    adj <- 0.0
  }

  return(adj)
}

#' srr_stats
#' @srrstats {G2.1a} Ensures all covariance adjustments align with the input model assumptions.
#' @srrstats {G3.1a} Accounts for adjustments based on finite population corrections and weak exogeneity assumptions.
#' @srrstats {G5.2a} Provides meaningful warnings or messages for invalid covariance settings or assumptions.
#' @noRd
NULL

apes_adjust_covariance_ <- function(
    v, delta, gamma, k_list, adj, sampling_fe,
    weak_exo, panel_structure) {
  if (adj > 0.0) {
    # Simplify covariance if sampling assumptions are imposed
    if (sampling_fe == "independence") {
      v <- v + adj * group_sums_var_(delta, k_list[[1L]])
      if (length(k_list) > 1L) {
        v <- v + adj * (group_sums_var_(delta, k_list[[2L]]) - crossprod(delta))
      }
      if (panel_structure == "network" && length(k_list) > 2L) {
        v <- v + adj * (group_sums_var_(delta, k_list[[3L]]) - crossprod(delta))
      }
    }

    # Add covariance in case of weak exogeneity
    if (weak_exo) {
      if (panel_structure == "classic") {
        cl <- group_sums_cov_(delta, gamma, k_list[[1L]])
        v <- v + adj * (cl + t(cl))
        rm(cl)
      } else if (length(k_list) > 2L) {
        cl <- group_sums_cov_(delta, gamma, k_list[[3L]])
        v <- v + adj * (cl + t(cl))
        rm(cl)
      }
    }
  }
  return(v)
}

#' srr_stats
#' @srrstats {G2.1a} Validates analytical bias correction computations against model assumptions.
#' @srrstats {G3.1a} Handles bias correction across panel structures (`classic` or `network`) and varying numbers of fixed effects.
#' @srrstats {RE5.0} Scales bias correction computations to handle large panels efficiently.
#' @srrstats {G5.2a} Issues clear errors for unsupported bias correction settings or invalid assumptions.
#' @noRd
NULL

apes_bias_correction_ <- function(
    eta, family, x, beta, binary, nt, p, ppsi,
    z, w, k_list, panel_structure, l, k, mpsi, v) {
  # Compute second-order partial derivatives
  delta2 <- matrix(NA_real_, nt, p)
  delta2[, !binary] <- partial_mu_eta_(eta, family, 3L)
  for (i in seq.int(p)) {
    if (binary[[i]]) {
      eta0 <- eta - x[, i] * beta[[i]]
      delta2[, i] <- partial_mu_eta_(eta0 + beta[[i]], family, 2L) -
        partial_mu_eta_(eta0, family, 2L)
      rm(eta0)
    } else {
      delta2[, i] <- beta[[i]] * delta2[, i]
    }
  }

  # Compute bias terms for requested bias correction
  if (panel_structure == "classic") {
    # Compute \hat{B} and \hat{D}
    b <- group_sums_(delta2 + ppsi * z, w, k_list[[1L]]) / (2.0 * nt)
    if (k > 1L) {
      b <- (b + group_sums_(delta2 + ppsi * z, w, k_list[[2L]])) / (2.0 * nt)
    }

    # Compute spectral density part of \hat{B}
    if (l > 0L) {
      b <- (b - group_sums_spectral_(mpsi * w, v, w, l, k_list[[1L]])) / nt
    }
  } else {
    # Compute \hat{D}_{1}, \hat{D}_{2}, and \hat{B}
    b <- group_sums_(delta2 + ppsi * z, w, k_list[[1L]]) / (2.0 * nt)
    b <- (b + group_sums_(delta2 + ppsi * z, w, k_list[[2L]])) / (2.0 * nt)
    if (k > 2L) {
      b <- (b + group_sums_(delta2 + ppsi * z, w, k_list[[3L]])) / (2.0 * nt)
    }

    # Compute spectral density part of \hat{B}
    if (k > 2L && l > 0L) {
      b <- (b - group_sums_spectral_(mpsi * w, v, w, l, k_list[[3L]])) / nt
    }
  }
  rm(delta2)

  return(b)
}
