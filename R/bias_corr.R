#' srr_stats
#' @srrstats {G1.0} The implementation is aligned with established methods, including those described in Stammann (2018), Fernández-Val and Weidner (2016), and others.
#' @srrstats {G2.1a} Ensures input objects are of the expected class (`feglm`).
#' @srrstats {G2.3a} Validates string arguments like `panel_structure` using `match.arg()` for predefined values.
#' @srrstats {G2.14a} Provides errors for missing or invalid inputs, such as non-`feglm` objects.
#' @srrstats {G2.14b} Provides clear error messages when the data structure is incompatible with the model requirements.
#' @srrstats {G3.1a} Supports structured panels (`classic` or `network`) for analyzing fixed effects.
#' @srrstats {RE5.0} Efficient handling of computational scaling for large panels through fixed-effect groupings.
#' @srrstats {G5.2a} Produces unique and informative error messages for all stopping conditions.
#' @srrstats {G5.4a} Includes logical checks for computational edge cases, such as unsupported models.
#' @noRd
NULL

#' NA_standards
#' @srrstatsNA {G2.14} Missing observations are dropped, otherwise providing imputation methods would bias the estimation (i.e., replacing all missing values with the median).
#' @noRd
NULL

#' @title Asymptotic bias correction after fitting binary choice models with a
#'  1,2,3-way error component
#'
#' @description Post-estimation routine to substantially reduce the incidental
#'  parameter bias problem. Applies the analytical bias correction derived by
#'  Fernández-Val and Weidner (2016) and Hinz, Stammann, and Wanner (2020) to
#'  obtain bias-corrected estimates of the structural parameters and is
#'  currently restricted to \code{\link[stats]{binomial}} with 1,2,3-way fixed
#'  effects.
#'
#' @param object an object of class \code{"feglm"}.
#' @param l unsigned integer indicating a bandwidth for the estimation of
#'  spectral densities proposed by Hahn and Kuersteiner (2011). The default is
#'  zero, which should be used if all regressors are assumed to be strictly
#'  exogenous with respect to the idiosyncratic error term. In the presence of
#'  weakly exogenous regressors, e.g. lagged outcome variables, we suggest to
#'  choose a bandwidth between one and four. Note that the order of factors to
#'  be partialed out is important for bandwidths larger than zero.
#' @param panel_structure a string equal to \code{"classic"} or \code{"network"}
#'  which determines the structure of the panel used. \code{"classic"} denotes
#'  panel structures where for example the same cross-sectional units are
#'  observed several times (this includes pseudo panels). \code{"network"}
#'  denotes panel structures where for example bilateral trade flows are
#'  observed for several time periods. Default is \code{"classic"}.
#'
#' @return A named list of classes \code{"bias_corr"} and \code{"feglm"}.
#'
#' @references Czarnowske, D. and A. Stammann (2020). "Fixed Effects Binary
#'  Choice Models: Estimation and Inference with Long Panels". ArXiv e-prints.
#' @references Fernández-Val, I. and M. Weidner (2016). "Individual and time
#'  effects in nonlinear panel models with large N, T". Journal of Econometrics,
#'  192(1), 291-312.
#' @references Fernández-Val, I. and M. Weidner (2018). "Fixed effects
#'  estimation of large-t panel data models". Annual Review of Economics, 10,
#'  109-138.
#' @references Hahn, J. and G. Kuersteiner (2011). "Bias reduction for dynamic
#'  nonlinear panel models with fixed effects". Econometric Theory, 27(6),
#'  1152-1191.
#' @references Hinz, J., A. Stammann, and J. Wanner (2020). "State Dependence
#'  and Unobserved Heterogeneity in the Extensive Margin of Trade". ArXiv
#'  e-prints.
#' @references Neyman, J. and E. L. Scott (1948). "Consistent estimates based on
#'  partially consistent observations". Econometrica, 16(1), 1-32.
#'
#' @seealso \code{\link{feglm}}
#'
#' @examples
#' # subset trade flows to avoid fitting time warnings during check
#' set.seed(123)
#' trade_2006 <- trade_panel[trade_panel$year == 2006, ]
#' trade_2006 <- trade_2006[sample(nrow(trade_2006), 500), ]
#'
#' trade_2006$trade <- ifelse(trade_2006$trade > 100, 1L, 0L)
#'
#' # Fit 'feglm()'
#' mod <- feglm(trade ~ lang | year, trade_2006, family = binomial())
#'
#' # Apply analytical bias correction
#' mod_bc <- bias_corr(mod)
#' summary(mod_bc)
#'
#' @export
bias_corr <- function(
    object = NULL,
    l = 0L,
    panel_structure = c("classic", "network")) {
  # Check validity of 'object'
  apes_bias_check_object_(object, fun = "bias_corr")

  # Check validity of 'panel_structure'
  panel_structure <- match.arg(panel_structure)

  # Extract model information
  beta_uncorr <- object[["coefficients"]]
  control <- object[["control"]]
  data <- object[["data"]]
  family <- object[["family"]]
  formula <- object[["formula"]]
  lvls_k <- object[["lvls_k"]]
  nms_sp <- names(beta_uncorr)
  nt <- object[["nobs"]][["nobs"]]
  k_vars <- names(lvls_k)
  k <- length(lvls_k)

  # Check if binary choice model
  apes_bias_check_binary_model_(family, fun = "bias_corr")

  # Check if the number of FEs is > 3
  bias_corr_check_fixed_effects_(k)

  # Check if provided object matches requested panel structure
  apes_bias_check_panel_(panel_structure, k)

  # Extract model response, regressor matrix, and weights
  y <- data[[1L]]
  x <- model.matrix(formula, data, rhs = 1L)[, -1L, drop = FALSE]
  attr(x, "dimnames") <- NULL
  wt <- object[["weights"]]

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
    x <- object[["mx"]]
  } else {
    x <- center_variables_r_(x, w, k_list, control[["center_tol"]], control[["iter_max"]], control[["iter_interrupt"]])
  }

  # Compute bias terms for requested bias correction
  if (panel_structure == "classic") {
    # Compute \hat{B} and \hat{D}
    b <- as.vector(group_sums_(x * z, w, k_list[[1L]])) / 2.0 / nt
    if (k > 1L) {
      b <- b + as.vector(group_sums_(x * z, w, k_list[[2L]])) / 2.0 / nt
    }

    # Compute spectral density part of \hat{B}
    if (l > 0L) {
      b <- (b + group_sums_spectral_(x * w, v, w, l, k_list[[1L]])) / nt
    }
  } else {
    # Compute \hat{D}_{1}, \hat{D}_{2}, and \hat{B}
    b <- group_sums_(x * z, w, k_list[[1L]]) / (2.0 * nt)
    b <- (b + group_sums_(x * z, w, k_list[[2L]])) / (2.0 * nt)
    if (k > 2L) {
      b <- (b + group_sums_(x * z, w, k_list[[3L]])) / (2.0 * nt)
    }

    # Compute spectral density part of \hat{B}
    if (k > 2L && l > 0L) {
      b <- (b + group_sums_spectral_(x * w, v, w, l, k_list[[3L]])) / nt
    }
  }

  # Compute bias-corrected structural parameters
  beta <- beta_uncorr - solve(object[["hessian"]] / nt, b)
  names(beta) <- nms_sp

  # Update \eta and first- and second-order derivatives
  eta <- feglm_offset_(object, x %*% beta)
  mu <- family[["linkinv"]](eta)
  mu_eta <- family[["mu.eta"]](eta)
  v <- wt * (y - mu)
  w <- wt * mu_eta
  if (family[["link"]] != "logit") {
    h <- mu_eta / family[["variance"]](mu)
    v <- h * v
    w <- h * w
    rm(h)
  }

  # Update centered regressor matrix
  x <- center_variables_r_(x, w, k_list, control[["center_tol"]], control[["iter_max"]], control[["iter_interrupt"]])
  colnames(x) <- nms_sp

  # Update hessian
  h <- crossprod(x * sqrt(w))
  dimnames(h) <- list(nms_sp, nms_sp)

  # Update result list
  object[["coefficients"]] <- beta
  object[["eta"]] <- eta
  if (control[["keep_mx"]]) object[["mx"]] <- x
  object[["hessian"]] <- h
  object[["coefficients_uncorr"]] <- beta_uncorr
  object[["bias_term"]] <- b
  object[["panel_structure"]] <- panel_structure
  object[["bandwidth"]] <- l

  # Add additional class to result list
  attr(object, "class") <- c("feglm", "bias_corr")

  # Return updated list
  object
}

bias_corr_check_fixed_effects_ <- function(lvls_k) {
  if (length(lvls_k) > 3) {
    stop(
      "bias_corr() only supports models with up to three-way fixed effects.",
      call. = FALSE
    )
  }
}
