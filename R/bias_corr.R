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
#' @param L unsigned integer indicating a bandwidth for the estimation of
#'  spectral densities proposed by Hahn and Kuersteiner (2011). The default is
#'  zero, which should be used if all regressors are assumed to be strictly
#'  exogenous with respect to the idiosyncratic error term. In the presence of
#'  weakly exogenous regressors, e.g. lagged outcome variables, we suggest to
#'  choose a bandwidth between one and four. Note that the order of factors to
#'  be partialed out is important for bandwidths larger than zero.
#' @param panel.structure a string equal to \code{"classic"} or \code{"network"}
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
#' trade_short <- trade_panel[trade_panel$year %in% 2002L:2006L, ]
#' trade_short$trade <- ifelse(trade_short$trade > 100, 1L, 0L)
#'
#' # Fit 'feglm()'
#' mod <- feglm(trade ~ lang | year, trade_short, family = binomial())
#'
#' # Apply analytical bias correction
#' mod_bc <- bias_corr(mod)
#' summary(mod_bc)
#'
#' @export
bias_corr <- function(
    object = NULL,
    L = 0L,
    panel.structure = c("classic", "network")) {
  # Check validity of 'object'
  if (is.null(object)) {
    stop("'object' has to be specified.", call. = FALSE)
  } else if (!inherits(object, "feglm")) {
    stop("'bias_corr' called on a non-'feglm' object.", call. = FALSE)
  }

  # Check validity of 'panel.structure'
  panel.structure <- match.arg(panel.structure)

  # Extract model information
  beta.uncorr <- object[["coefficients"]]
  control <- object[["control"]]
  data <- object[["data"]]
  eps <- .Machine[["double.eps"]]
  family <- object[["family"]]
  formula <- object[["formula"]]
  lvls.k <- object[["lvls.k"]]
  nms.sp <- names(beta.uncorr)
  nt <- object[["nobs"]][["nobs"]]
  k.vars <- names(lvls.k)
  k <- length(lvls.k)

  # Check if binary choice model
  if (family[["family"]] != "binomial") {
    stop(
      "'bias_corr' currently only supports binary choice models.",
      call. = FALSE
    )
  }

  # Check if the number of FEs is > 3
  if (length(lvls.k) > 3) {
    stop(
      "bias_corr() only supports models with up to three-way fixed effects.",
      call. = FALSE
    )
  }

  # Check if provided object matches requested panel structure
  if (panel.structure == "classic") {
    if (!(k %in% c(1L, 2L))) {
      stop(
        paste(
          "panel.structure == 'classic' expects a one- or two-way fixed",
          "effect model."
        ),
        call. = FALSE
      )
    }
  } else {
    if (!(k %in% c(2L, 3L))) {
      stop(
        paste(
          "panel.structure == 'network' expects a two- or three-way fixed",
          "effects model."
        ),
        call. = FALSE
      )
    }
  }

  # Extract model response, regressor matrix, and weights
  y <- data[[1L]]
  X <- model.matrix(formula, data, rhs = 1L)[, -1L, drop = FALSE]
  attr(X, "dimnames") <- NULL
  wt <- object[["weights"]]

  # Generate auxiliary list of indexes for different sub panels
  k.list <- get_index_list_(k.vars, data)

  # Compute derivatives and weights
  eta <- object[["eta"]]
  mu <- family[["linkinv"]](eta)
  mu.eta <- family[["mu.eta"]](eta)
  v <- wt * (y - mu)
  w <- wt * mu.eta
  z <- wt * partial_mu_eta_(eta, family, 2L)
  if (family[["link"]] != "logit") {
    h <- mu.eta / family[["variance"]](mu)
    v <- h * v
    w <- h * w
    z <- h * z
    rm(h)
  }

  # Center regressor matrix (if required)
  if (control[["keep.mx"]]) {
    MX <- object[["MX"]]
  } else {
    MX <- center_variables_(X, NA_real_, w, k.list, control[["center.tol"]], 100000L, FALSE)
  }

  # Compute bias terms for requested bias correction
  if (panel.structure == "classic") {
    # Compute \hat{B} and \hat{D}
    b <- as.vector(group_sums_(MX * z, w, k.list[[1L]])) / 2.0 / nt
    if (k > 1L) {
      b <- b + as.vector(group_sums_(MX * z, w, k.list[[2L]])) / 2.0 / nt
    }

    # Compute spectral density part of \hat{B}
    if (L > 0L) {
      b <- (b + group_sums_spectral_(MX * w, v, w, L, k.list[[1L]])) / nt
    }
  } else {
    # Compute \hat{D}_{1}, \hat{D}_{2}, and \hat{B}
    b <- group_sums_(MX * z, w, k.list[[1L]]) / (2.0 * nt)
    b <- (b + group_sums_(MX * z, w, k.list[[2L]])) / (2.0 * nt)
    if (k > 2L) {
      b <- (b + group_sums_(MX * z, w, k.list[[3L]])) / (2.0 * nt)
    }

    # Compute spectral density part of \hat{B}
    if (k > 2L && L > 0L) {
      b <- (b + group_sums_spectral_(MX * w, v, w, L, k.list[[3L]])) / nt
    }
  }

  # Compute bias-corrected structural parameters
  beta <- solve_bias_(beta.uncorr, object[["Hessian"]], nt, -b)
  names(beta) <- nms.sp

  # Update \eta and first- and second-order derivatives
  eta <- feglm_offset_(object, solve_y_(X, beta))
  mu <- family[["linkinv"]](eta)
  mu.eta <- family[["mu.eta"]](eta)
  v <- wt * (y - mu)
  w <- wt * mu.eta
  if (family[["link"]] != "logit") {
    h <- mu.eta / family[["variance"]](mu)
    v <- h * v
    w <- h * w
    rm(h)
  }

  # Update centered regressor matrix
  MX <- center_variables_(X, NA_real_, w, k.list, control[["center.tol"]], 100000L, FALSE)
  colnames(MX) <- nms.sp

  # Update Hessian
  H <- crossprod_(MX, w, TRUE, TRUE)
  dimnames(H) <- list(nms.sp, nms.sp)

  # Update result list
  object[["coefficients"]] <- beta
  object[["eta"]] <- eta
  if (control[["keep.mx"]]) object[["MX"]] <- MX
  object[["Hessian"]] <- H
  object[["coefficients.uncorr"]] <- beta.uncorr
  object[["bias.term"]] <- b
  object[["panel.structure"]] <- panel.structure
  object[["bandwidth"]] <- L

  # Add additional class to result list
  attr(object, "class") <- c("feglm", "bias_corr")

  # Return updated list
  object
}
