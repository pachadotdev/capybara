#' srr_stats
#' @srrstats {G1.0} Implements covariance matrix extraction methods for `apes`, `feglm`, and `felm` objects.
#' @srrstats {G2.1a} Validates input objects as instances of `apes`, `feglm`, or `felm`.
#' @srrstats {G2.2} Provides various covariance estimation types including `hessian`, `outer.product`, and `sandwich`.
#' @srrstats {G2.3} Handles cases with or without clustering variables, ensuring flexibility for diverse use cases.
#' @srrstats {G3.0} Handles edge cases such as non-invertible hessians or missing cluster variables gracefully with informative errors.
#' @srrstats {G4.0} Integrates seamlessly with the modeling pipeline, supporting consistent outputs for downstream analysis.
#' @srrstats {RE2.1} Ensures compatibility with multiway clustering approaches as proposed by Cameron, Gelbach, and Miller (2011).
#' @srrstats {RE2.3} Supports computation of robust covariance estimates for generalized linear models and linear models.
#' @srrstats {RE5.0} Ensures that the output covariance matrix is correctly labeled for interpretability.
#' @srrstats {RE5.2} Provides explicit errors for invalid or missing clustering variables in clustered covariance computation.
#' @srrstats {RE6.1} Implements efficient matrix operations to handle large-scale data and high-dimensional models.
#' @noRd
NULL

#' @title Covariance matrix for APEs
#'
#' @description Covariance matrix for the estimator of the
#'  average partial effects from objects returned by \code{\link{apes}}.
#'
#' @param object an object of class \code{"apes"}.
#' @param ... additional arguments.
#'
#' @return A named matrix of covariance estimates.
#'
#' @seealso \code{\link{apes}}
#'
#' @export
#'
#' @noRd
vcov.apes <- function(object, ...) {
  object[["vcov"]]
}

#' @title Covariance matrix for GLMs
#'
#' @description Covariance matrix for the estimator of the structural parameters
#'  from objects returned by \code{\link{feglm}}. The covariance is computed
#'  during model fitting - either the inverse Hessian (default) or the
#'  sandwich estimator if cluster variables are specified in the formula.
#'
#' @param object an object of class \code{"feglm"}.
#' @param ... additional arguments (currently ignored).
#'
#' @return A named matrix of covariance estimates.
#'
#' @references Cameron, C., J. Gelbach, and D. Miller (2011). "Robust Inference
#'  With Multiway Clustering". Journal of Business & Economic Statistics 29(2).
#'
#' @seealso \code{\link{feglm}}
#'
#' @examples
#' # Model without clustering - returns inverse Hessian covariance
#' mod <- fepoisson(mpg ~ wt | cyl, mtcars)
#' round(vcov(mod), 5)
#'
#' # Model with clustering - returns sandwich covariance
#' mod_cl <- fepoisson(mpg ~ wt | cyl | am, mtcars)
#' round(vcov(mod_cl), 5)
#'
#' @return A named matrix of covariance estimates.
#'
#' @export
vcov.feglm <- function(object, ...) {
  v <- object[["vcov"]]

  # Check if vcov exists

  if (is.null(v)) {
    stop("Covariance matrix not found in model object.", call. = FALSE)
  }

  # Add names to match coefficients
  nms <- names(object[["coefficients"]])
  if (!is.null(nms) && length(nms) > 0) {
    # Handle NA coefficients (collinear)
    non_na <- !is.na(object[["coefficients"]])
    nms <- nms[non_na]
    if (length(nms) == nrow(v)) {
      dimnames(v) <- list(nms, nms)
    }
  }

  v
}

vcov_feglm_vars_ <- function(object) {
  suppressWarnings({
    attr(terms(object[["formula"]], rhs = 3L), "term.labels")
  })
}

vcov_feglm_hessian_covariance_ <- function(h, p) {
  v <- try(solve(h), silent = TRUE)
  if (inherits(v, "try-error")) {
    v <- matrix(Inf, p, p)
  }
  v
}

vcov_feglm_outer_covariance_ <- function(g, p) {
  v <- try(solve(crossprod(g)), silent = TRUE)
  if (inherits(v, "try-error")) {
    v <- matrix(Inf, p, p)
  }
  v
}

vcov_feglm_covmat_ <- function(
  object, type, h, g,
  cl_vars, k, p
) {
  # Check if the hessian is invertible and compute its inverse
  v <- try(solve(h), silent = TRUE)
  if (inherits(v, "try-error")) {
    v <- matrix(Inf, p, p)
  } else {
    # Compute clustered covariance (fallback when precomputed not available)
    if (isFALSE(k >= 1L)) {
      vcov_feglm_cluster_nocluster_()
    }
    d <- vcov_feglm_cluster_data_(object, cl_vars)
    d[cl_vars] <- lapply(d[cl_vars], check_factor_)
    sp_vars <- colnames(g)
    g <- cbind(d, g)
    rm(d)
    b <- vcov_feglm_clustered_cov_(g, cl_vars, sp_vars, p)
    # Sandwich formula: bread %*% meat %*% bread
    v <- v %*% b %*% v
  }

  # Return covariance estimate
  v
}

vcov_feglm_cluster_nocluster_ <- function() {
  stop(
    paste(
      "No cluster variable was found.",
      "Please specify a cluster variable",
      "in the model formula."
    ),
    call. = FALSE
  )
}

vcov_feglm_cluster_data_ <- function(object, cl_vars, model = "feglm") {
  d <- try(object[["data"]][, cl_vars, drop = FALSE], silent = TRUE)
  if (inherits(d, "try-error")) {
    vcov_feglm_cluster_notfound_(model)
  }
  d
}

vcov_feglm_cluster_notfound_ <- function(model) {
  stop(
    paste0(
      "At least one cluster variable was not found. ",
      "Ensure to pass variables that are not part of the model ",
      "itself, but are required to compute clustered standard errors ",
      "to '", model, "'. This can be done via 'formula'. See documentation",
      "for details."
    ),
    call. = FALSE
  )
}

# Ensure cluster variables are factors ----

vcov_feglm_clustered_cov_ <- function(g, cl_vars, sp_vars, p) {
  # Multiway clustering by Cameron, Gelbach, and Miller (2011)
  b <- Reduce(function(acc, i) {
    # Generate all combinations of clustering variables
    cl_combn <- combn(cl_vars, i, simplify = FALSE)

    br <- Reduce(function(acc_inner, cl) {
      # Compute sum within each cluster (base R aggregate)
      grouped_data <- stats::aggregate(g[sp_vars], by = g[cl], FUN = function(x) sum(x, na.rm = TRUE))

      # Compute crossproduct, dropping clustering columns
      acc_inner + crossprod(as.matrix(grouped_data[sp_vars]))
    }, cl_combn, init = matrix(0.0, p, p))

    # Alternating sign adjustment
    if (i %% 2L) acc + br else acc - br
  }, seq_along(cl_vars), init = matrix(0.0, p, p))

  return(b)
}

# Particular case for linear models ----

#' @title Covariance matrix for LMs
#'
#' @description Covariance matrix for the estimator of the structural parameters
#'  from objects returned by \code{\link{felm}}. The covariance is computed
#'  from the hessian, the scores, or a combination of both after convergence.
#'
#' @param object an object of class \code{"felm"}.
#' @param type the type of covariance estimate required. \code{"hessian"} refers
#'  to the inverse of the negative expected hessian after convergence and is the
#'  default option. \code{"outer.product"} is the outer-product-of-the-gradient
#'  estimator. \code{"sandwich"} computes a clustered covariance matrix
#'  (sandwich estimator) given some cluster variables specified in the formula.
#'
#' @param ... additional arguments.
#'
#' @return A named matrix of covariance estimates.
#'
#' @seealso \code{\link{felm}}
#'
#' @examples
#' # same as the example in felm but extracting the covariance matrix
#' mod <- felm(log(mpg) ~ log(wt) | cyl | am, mtcars)
#' vcov(mod, type = "sandwich")
#'
#' @export
vcov.felm <- function(
  object,
  type = c("hessian", "outer.product", "sandwich"),
  ...
) {
  # Check validity of input argument 'type'
  type <- match.arg(type)

  # Extract cluster from formula
  # it is totally fine not to have a cluster variable
  cl_vars <- vcov_felm_vars_(object)
  k <- length(cl_vars)

  # Compute requested type of covariance matrix
  h <- object[["hessian"]]
  p <- ncol(h)

  if (type == "hessian") {
    # If the hessian is invertible, compute its inverse
    v <- vcov_feglm_hessian_covariance_(h, p)
  } else {
    g <- get_score_matrix_felm_(object)
    if (type == "outer.product") {
      # Check if the OP is invertible and compute its inverse
      v <- vcov_feglm_outer_covariance_(g, p)
    } else {
      v <- vcov_felm_covmat_(
        object, type, h, g,
        cl_vars, k, p
      )
    }
  }

  v
}

vcov_felm_vars_ <- function(object) {
  suppressWarnings({
    attr(terms(object[["formula"]], rhs = 3L), "term.labels")
  })
}

vcov_felm_covmat_ <- function(
  object, type, h, g,
  cl_vars, k, p
) {
  # Check if the hessian is invertible and compute its inverse
  v <- try(solve(h), silent = TRUE)
  if (inherits(v, "try-error")) {
    v <- matrix(Inf, p, p)
  } else {
    # Compute clustered covariance (sandwich estimator)
    if (isFALSE(k >= 1L)) {
      vcov_feglm_cluster_nocluster_()
    }
    d <- vcov_feglm_cluster_data_(object, cl_vars, "felm")
    d[cl_vars] <- lapply(d[cl_vars], check_factor_)
    sp_vars <- colnames(g)
    g <- cbind(d, g)
    rm(d)
    b <- vcov_feglm_clustered_cov_(g, cl_vars, sp_vars, p)
    # Sandwich formula: bread %*% meat %*% bread
    v <- v %*% b %*% v
  }

  # Return covariance estimate
  v
}
