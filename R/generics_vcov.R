#' srr_stats
#' @srrstats {G1.0} Implements covariance matrix extraction methods for `apes`, `feglm`, and `felm` objects.
#' @srrstats {G2.1a} Validates input objects as instances of `apes`, `feglm`, or `felm`.
#' @srrstats {G2.2} Provides various covariance estimation types including `hessian`, `outer.product`, `sandwich`, and `clustered`.
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
#' from the hessian, the scores, or a combination of both after convergence.
#'
#' @param object an object of class \code{"feglm"}.
#' @param type the type of covariance estimate required. \code{"hessian"} refers
#'  to the inverse of the negative expected hessian after convergence and is the
#'  default option. \code{"outer.product"} is the outer-product-of-the-gradient
#'  estimator. \code{"sandwich"} is the sandwich estimator (sometimes also
#'  referred as robust estimator), and \code{"clustered"} computes a clustered
#'  covariance matrix given some cluster variables.
#'
#' @param ... additional arguments.
#'
#' @return A named matrix of covariance estimates.
#'
#' @references Cameron, C., J. Gelbach, and D. Miller (2011). "Robust Inference
#'  With Multiway Clustering". Journal of Business & Economic Statistics 29(2).
#'
#' @seealso \code{\link{feglm}}
#'
#' @examples
#' # same as the example in feglm but extracting the covariance matrix
#'
#' # subset trade flows to avoid fitting time warnings during check
#' set.seed(123)
#' trade_2006 <- trade_panel[trade_panel$year == 2006, ]
#' trade_2006 <- trade_2006[sample(nrow(trade_2006), 500), ]
#'
#' mod <- fepoisson(
#'   trade ~ log_dist + lang + cntg + clny | exp_year + imp_year | pair,
#'   trade_2006
#' )
#'
#' round(vcov(mod, type = "clustered"), 5)
#'
#' @return A named matrix of covariance estimates.
#'
#' @export
vcov.feglm <- function(
    object,
    type = c("hessian", "outer.product", "sandwich", "clustered"),
    ...) {
  # Check validity of input argument 'type'
  type <- match.arg(type)

  # Extract cluster from formula
  # it is totally fine not to have a cluster variable
  cl_vars <- vcov_feglm_vars_(object)
  k <- length(cl_vars)

  if (isTRUE(k >= 1L) && type != "clustered") {
    type <- "clustered"
  }

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
      v <- vcov_feglm_covmat_(
        object, type, h, g,
        cl_vars, k, p
      )
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
  v <- try(solve(g), silent = TRUE)
  if (inherits(v, "try-error")) {
    v <- matrix(Inf, p, p)
  }
  v
}

vcov_feglm_covmat_ <- function(
    object, type, h, g,
    cl_vars, k, p) {
  # Check if the hessian is invertible and compute its inverse
  v <- try(solve(h), silent = TRUE)
  if (inherits(v, "try-error")) {
    v <- matrix(Inf, p, p)
  } else {
    # Compute inner part of the sandwich formula
    if (type == "sandwich") {
      b <- crossprod(g)
    } else {
      if (isFALSE(k >= 1L)) {
        vcov_feglm_cluster_nocluster_()
      }
      d <- vcov_feglm_cluster_data_(object, cl_vars)
      d[, (cl_vars) := lapply(.SD, check_factor_), .SDcols = cl_vars]
      sp_vars <- colnames(g)
      g <- cbind(d, g)
      rm(d)
      b <- vcov_feglm_clustered_cov_(g, cl_vars, sp_vars, p)
    }
    # Sandwich formula
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
  d <- try(object[["data"]][, .SD, .SDcols = cl_vars], silent = TRUE)
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
  b <- matrix(0.0, p, p)

  for (i in seq_along(cl_vars)) {
    # Generate all combinations of clustering variables
    cl_combn <- combn(cl_vars, i, simplify = FALSE)

    br <- matrix(0.0, p, p)

    for (cl in cl_combn) {
      # Compute sum within each cluster
      grouped_data <- g[, lapply(.SD, sum), by = cl, .SDcols = sp_vars]

      # Compute crossproduct, dropping clustering columns
      br <- br + crossprod(as.matrix(grouped_data[, .SD, .SDcols = sp_vars]))
    }

    # Alternating sign adjustment
    b <- if (i %% 2L) b + br else b - br
  }

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
#'  estimator. \code{"sandwich"} is the sandwich estimator (sometimes also
#'  referred as robust estimator), and \code{"clustered"} computes a clustered
#'  covariance matrix given some cluster variables.
#'
#' @param ... additional arguments.
#'
#' @return A named matrix of covariance estimates.
#'
#' @seealso \code{\link{felm}}
#'
#' @examples
#' # same as the example in felm but extracting the covariance matrix
#'
#' # subset trade flows to avoid fitting time warnings during check
#' set.seed(123)
#' trade_2006 <- trade_panel[trade_panel$year == 2006, ]
#' trade_2006 <- trade_2006[sample(nrow(trade_2006), 500), ]
#'
#' mod <- felm(
#'   trade ~ log_dist + lang + cntg + clny | exp_year + imp_year | pair,
#'   trade_2006
#' )
#'
#' round(vcov(mod, type = "clustered"), 5)
#'
#' @export
vcov.felm <- function(
    object,
    type = c("hessian", "outer.product", "sandwich", "clustered"),
    ...) {
  # Check validity of input argument 'type'
  type <- match.arg(type)

  # Extract cluster from formula
  # it is totally fine not to have a cluster variable
  cl_vars <- vcov_felm_vars_(object)
  k <- length(cl_vars)

  if (isTRUE(k >= 1L) && type != "clustered") {
    type <- "clustered"
  }

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
    cl_vars, k, p) {
  # Check if the hessian is invertible and compute its inverse
  v <- try(solve(h), silent = TRUE)
  if (inherits(v, "try-error")) {
    v <- matrix(Inf, p, p)
  } else {
    # Compute inner part of the sandwich formula
    if (type == "sandwich") {
      b <- crossprod(g)
    } else {
      if (isFALSE(k >= 1L)) {
        vcov_feglm_cluster_nocluster_()
      }
      d <- vcov_feglm_cluster_data_(object, cl_vars, "felm")
      d[, (cl_vars) := lapply(.SD, check_factor_), .SDcols = cl_vars]
      sp_vars <- colnames(g)
      g <- cbind(d, g)
      rm(d)
      b <- vcov_feglm_clustered_cov_(g, cl_vars, sp_vars, p)
    }
    # Sandwich formula
    v <- v %*% b %*% v
  }

  # Return covariance estimate
  v
}
