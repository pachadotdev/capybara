# srr_stats
# {G1.0} Implements covariance matrix extraction methods for `apes`, `feglm`, and `felm` objects.
# {G2.1a} Validates input objects as instances of `apes`, `feglm`, or `felm`.
# {G2.2} Provides various covariance estimation types including `hessian`, `outer.product`, and `sandwich`.
# {G2.3} Handles cases with or without clustering variables, ensuring flexibility for diverse use cases.
# {G3.0} Handles edge cases such as non-invertible hessians or missing cluster variables gracefully with informative errors.
# {G4.0} Integrates seamlessly with the modeling pipeline, supporting consistent outputs for downstream analysis.
# {RE2.1} Ensures compatibility with multiway clustering approaches as proposed by Cameron, Gelbach, and Miller (2011).
# {RE2.3} Supports computation of robust covariance estimates for generalized linear models and linear models.
# {RE5.0} Ensures that the output covariance matrix is correctly labeled for interpretability.
# {RE5.2} Provides explicit errors for invalid or missing clustering variables in clustered covariance computation.
# {RE6.1} Implements efficient matrix operations to handle large-scale data and high-dimensional models.

#' @title Covariance matrix for APEs
#'
#' @description Covariance matrix for the estimator of the average partial effects from objects returned by \link{apes}.
#'
#' @param object an object of class \code{"apes"}.
#' @param ... additional arguments.
#'
#' @return A named matrix of covariance estimates.
#'
#' @seealso \link{apes}
#'
#' @export
#'
#' @noRd
vcov.apes <- function(object, ...) {
  object[["vcov"]]
}

#' @title Covariance matrix for GLMs
#'
#' @description Covariance matrix for the estimator of the structural parameters from objects returned by \link{feglm}.
#'  The covariance is computed during model fitting - either the inverse Hessian (default) or the sandwich estimator if
#'  cluster variables are specified in the formula.
#'
#' @param object an object of class \code{"feglm"}.
#' @param ... additional arguments (currently ignored).
#'
#' @return A named matrix of covariance estimates.
#'
#' @references Cameron, C., J. Gelbach, and D. Miller (2011). "Robust Inference With Multiway Clustering". Journal of
#'  Business & Economic Statistics 29(2).
#'
#' @seealso \link{feglm}
#'
#' @examples
#' # Model with clustering - returns sandwich covariance
#' ross2004_s1 <- ross2004[ross2004$year == 1994, ]
#' ross2004_s1 <- ross2004_s1[ross2004_s1$ltrade >
#'   quantile(ross2004_s1$ltrade, 0.75), ]
#'
#' ross2004_s2 <- ross2004[ross2004$year == 1999, ]
#' ross2004_s2 <- ross2004_s2[ross2004_s2$ltrade >
#'   quantile(ross2004_s2$ltrade, 0.75), ]
#'
#' ross2004_subset <- rbind(ross2004_s1, ross2004_s2)
#'
#' fit <- fepoisson(ltrade ~ ldist + border | ctry1 | year, ross2004_subset)
#'
#' vcov(fit)
#'
#' @export
vcov.feglm <- function(object, ...) {
  v <- object[["vcov"]]

  if (is.null(v)) {
    stop("Covariance matrix not found in model object.", call. = FALSE)
  }

  # Add names to match coefficients
  coef_table <- object[["coef_table"]]
  nms <- rownames(coef_table)
  if (!is.null(nms) && length(nms) > 0) {
    dimnames(v) <- list(nms, nms)
  }

  v
}

# Particular case for linear models ----

#' @title Covariance matrix for LMs
#'
#' @description Covariance matrix for the estimator of the structural parameters from objects returned by \link{felm}.
#'  The covariance is computed during model fitting - either the inverse Hessian (default) or the sandwich estimator if
#'  cluster variables are specified in the formula.
#'
#' @param object an object of class \code{"felm"}.
#' @param ... additional arguments (currently ignored).
#'
#' @return A named matrix of covariance estimates.
#'
#' @references Cameron, C., J. Gelbach, and D. Miller (2011). "Robust Inference With Multiway Clustering". Journal of
#'  Business & Economic Statistics 29(2).
#'
#' @seealso \link{felm}
#'
#' @examples
#' # Model with clustering - returns sandwich covariance
#' ross2004_s1 <- ross2004[ross2004$year == 1994, ]
#' ross2004_s1 <- ross2004_s1[ross2004_s1$ltrade >
#'   quantile(ross2004_s1$ltrade, 0.75), ]
#'
#' ross2004_s2 <- ross2004[ross2004$year == 1999, ]
#' ross2004_s2 <- ross2004_s2[ross2004_s2$ltrade >
#'   quantile(ross2004_s2$ltrade, 0.75), ]
#'
#' ross2004_subset <- rbind(ross2004_s1, ross2004_s2)
#'
#' fit <- felm(ltrade ~ ldist | ctry1 | year, ross2004_subset)
#'
#' vcov(fit)
#'
#' @export
vcov.felm <- function(object, ...) {
  v <- object[["vcov"]]

  if (is.null(v)) {
    stop("Covariance matrix not found in model object.", call. = FALSE)
  }

  # Add names to match coefficients
  coef_table <- object[["coef_table"]]
  nms <- rownames(coef_table)
  if (!is.null(nms) && length(nms) > 0) {
    dimnames(v) <- list(nms, nms)
  }

  v
}

#' @title Recompute Sandwich Variance-Covariance Matrix
#'
#' @description Recompute the variance-covariance matrix for a fitted model using a different
#'  clustering structure or covariance type. This allows changing the vcov estimator without
#'  re-fitting the model, provided the model was fit with \code{keep_tx = TRUE} and
#'  \code{return_hessian = TRUE} in the control parameters.
#'
#' @param object a fitted model object of class \code{"feglm"} or \code{"felm"}.
#' @param cluster1 a vector or factor for the first clustering variable. Required for
#'  \code{"clustered"} and \code{"dyadic"} types.
#' @param cluster2 a vector or factor for the second clustering variable. Required for
#'  \code{"dyadic"} type.
#' @param type character string specifying the covariance type. One of:
#'  \itemize{
#'   \item \code{"hetero"}: heteroskedasticity-robust (no clustering, also known as \code{"HC0"})
#'   \item \code{"clustered"} or \code{"m-estimator"}: one-way cluster-robust
#'   \item \code{"dyadic"} or \code{"m-estimator-dyadic"}: dyadic cluster-robust for network/trade data
#'  }
#' @param ... additional arguments (currently ignored).
#'
#' @return A named matrix of covariance estimates.
#'
#' @details
#' The model must be fit with \code{fit_control(keep_tx = TRUE, return_hessian = TRUE)} to store
#' the centered design matrix, Hessian, and working residuals needed for vcov recomputation.
#' Note that \code{keep_data = TRUE} is NOT required - residuals are stored automatically when
#' \code{keep_tx = TRUE}.
#'
#' For dyadic clustering (used in gravity/trade models), \code{cluster1} and \code{cluster2}
#' represent the two entity dimensions (e.g., exporter and importer). The function handles
#' the full dyadic correlation structure including cross-entity correlations.
#'
#' @references
#' Cameron, C., J. Gelbach, and D. Miller (2011). "Robust Inference With Multiway Clustering".
#'  Journal of Business & Economic Statistics 29(2).
#'
#' Cameron, C. and D. Miller (2014). "Robust Inference for Dyadic Data". Unpublished manuscript.
#'
#' @seealso \link{feglm}, \link{felm}, \link{fit_control}
#'
#' @examples
#' # Refitting models
#'
#' ross2004_s1 <- ross2004[ross2004$year == 1994, ]
#' ross2004_s1 <- ross2004_s1[ross2004_s1$ltrade >
#'   quantile(ross2004_s1$ltrade, 0.75), ]
#'
#' ross2004_s2 <- ross2004[ross2004$year == 1999, ]
#' ross2004_s2 <- ross2004_s2[ross2004_s2$ltrade >
#'   quantile(ross2004_s2$ltrade, 0.75), ]
#'
#' ross2004_subset <- rbind(ross2004_s1, ross2004_s2)
#'
#' fepoisson(
#'   ltrade ~ ldist | ctry1, ross2004_subset,
#'   control = fit_control(vcov_type = "hetero")
#' )
#'
#' fepoisson(
#'   ltrade ~ ldist | ctry1, ross2004_subset,
#'   control = fit_control(vcov_type = "m-estimator")
#' )
#'
#' # Reusing models
#'
#' # Store required components
#' fit <- fepoisson(
#'   ltrade ~ ldist | ctry1, ross2004_subset,
#'   control = fit_control(keep_tx = TRUE, return_hessian = TRUE)
#' )
#'
#' # Heteroskedastic-robust HC0 sandwich (no cluster variable needed)
#' sandwich_vcov(fit, type = "hetero")
#'
#' #' One-way cluster
#' sandwich_vcov(fit, cluster1 = ross2004_subset$year, type = "clustered")
#'
#' @export
sandwich_vcov <- function(object, cluster1 = NULL, cluster2 = NULL,
                          type = c(
                            "hetero", "clustered", "m-estimator",
                            "dyadic", "m-estimator-dyadic"
                          ),
                          ...) {
  type <- match.arg(type)

  # Check required components
  MX <- object[["tx"]]
  H <- object[["hessian"]]

  if (is.null(MX)) {
    stop(
      "Centered design matrix (tx) not found. ",
      "Re-fit the model with keep_tx = TRUE in control parameters.",
      call. = FALSE
    )
  }

  if (is.null(H)) {
    stop(
      "Hessian matrix not found. ",
      "Re-fit the model with return_hessian = TRUE in control parameters.",
      call. = FALSE
    )
  }

  n_obs <- nrow(MX)

  # Get working residuals - prefer stored residuals (efficient, no keep_data needed)
  # Fall back to computing from data if stored residuals not available
  resid <- object[["working_residuals"]]

  if (is.null(resid) || length(resid) != n_obs) {
    # Fall back: try to compute from stored data
    mu <- object[["fitted_values"]]
    data <- object[["data"]]
    formula <- object[["formula"]]

    y <- NULL
    if (!is.null(data) && !is.null(formula)) {
      resp_var <- all.vars(formula)[1]
      if (resp_var %in% names(data)) {
        y <- data[[resp_var]]
      }
    }

    if (is.null(y) || is.null(mu) || length(y) != n_obs) {
      stop(
        "Working residuals not found. ",
        "Re-fit the model with keep_tx = TRUE (stores residuals automatically).",
        call. = FALSE
      )
    }

    resid <- as.numeric(y - mu)
  }

  # Subset cluster variables to match model observations if needed
  # data in fit object is already subset, so cluster vars should match n_obs
  if (!is.null(cluster1) && length(cluster1) != n_obs) {
    # Try to subset using .rownames
    rownames_used <- object[[".rownames"]]
    if (!is.null(rownames_used) && !is.null(names(cluster1))) {
      cluster1 <- cluster1[rownames_used]
    } else if (length(cluster1) > n_obs) {
      # Assume sequential subset
      warning("cluster1 length mismatch - using first n_obs elements")
      cluster1 <- cluster1[seq_len(n_obs)]
    }
  }

  if (!is.null(cluster2) && length(cluster2) != n_obs) {
    rownames_used <- object[[".rownames"]]
    if (!is.null(rownames_used) && !is.null(names(cluster2))) {
      cluster2 <- cluster2[rownames_used]
    } else if (length(cluster2) > n_obs) {
      warning("cluster2 length mismatch - using first n_obs elements")
      cluster2 <- cluster2[seq_len(n_obs)]
    }
  }

  # Call C++ function
  v <- compute_sandwich_vcov_(
    MX_r = MX,
    resid_r = resid,
    H_r = H,
    vcov_type = type,
    cluster1_r = cluster1,
    cluster2_r = cluster2
  )

  # Add names to match coefficients
  coef_table <- object[["coef_table"]]
  nms <- rownames(coef_table)
  if (!is.null(nms) && length(nms) > 0 && length(nms) == nrow(v)) {
    dimnames(v) <- list(nms, nms)
  }

  v
}
