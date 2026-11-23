#' @title Wild score bootstrap for Poisson models with fixed effects
#'
#' @description Unconstrained wild (score) bootstrap for the structural
#'  parameters of \link{fepoisson}, following Kline and Santos (2012).
#'  Non-linear models do not have well-defined residuals to perturb, so instead
#'  of perturbing the outcome, the estimated influence functions (scores) of the
#'  structural parameters are perturbed with random weights and mapped back to
#'  the parameters with the inverse Hessian of the full-sample fit. The model is
#'  estimated once, so there is no re-optimization in each draw.
#'
#' @inheritParams fepoisson
#' @param cluster an optional vector with one entry per row of \code{data}
#'  defining the clusters whose scores share a bootstrap weight. The default is
#'  one weight per observation.
#' @param B number of bootstrap draws.
#' @param weights_type distribution of the bootstrap weights, all with mean zero
#'  and unit variance: \code{"rademacher"} (default), \code{"mammen"}, or
#'  \code{"normal"}.
#' @param seed an optional integer passed to \code{set.seed()}.
#'
#' @details
#' With \eqn{s_g} the sum of the scores in cluster \eqn{g} (centered regressors
#' times working residuals), \eqn{H} the Hessian, and \eqn{W_g} i.i.d. weights,
#' each draw is \deqn{\hat\beta^* = \hat\beta + H^{-1} \sum_g W_g s_g.}
#' The covariance matrix of the draws converges to the cluster-robust sandwich
#' estimator.
#'
#' Only the structural parameters are perturbed. The fixed effects are held at
#' their full-sample estimates, so the draws are not suitable for statistics that
#' depend on the sampling variation of the fixed effects.
#'
#' @return A named list of class \code{"wild_bootstrap_fepoisson"} with
#'  \itemize{
#'   \item \code{fit}: the full-sample \code{"feglm"} fit.
#'   \item \code{coefficients}: the full-sample estimates.
#'   \item \code{draws}: a \code{B x p} matrix of bootstrap estimates.
#'   \item \code{vcov}: covariance matrix of the draws.
#'   \item \code{se}: standard errors from \code{vcov}.
#'   \item \code{ci}: centered percentile 95\% intervals
#'     (\eqn{2\hat\beta - q_{0.975}}, \eqn{2\hat\beta - q_{0.025}}).
#'  }
#'
#' @references Kline, P. and A. Santos (2012). "A Score Based Approach to Wild
#'  Bootstrap Inference". Journal of Econometric Methods 1(1).
#'
#' @seealso \link{fepoisson}, \link{sandwich_vcov}
#'
#' @examples
#' rose2004_subset <- rose2004[rose2004$year == 1999, ]
#' rose2004_subset <- rose2004_subset[rose2004_subset$ltrade >
#'   quantile(rose2004_subset$ltrade, 0.75), ]
#'
#' wild_bootstrap_fepoisson(
#'   ltrade ~ ldist | ctry1, rose2004_subset,
#'   cluster = rose2004_subset$ctry1, B = 99, seed = 1
#' )$se
#'
#' @export
wild_bootstrap_fepoisson <- function(
  formula = NULL,
  data = NULL,
  cluster = NULL,
  B = 999L,
  weights_type = c("rademacher", "mammen", "normal"),
  weights = NULL,
  offset = NULL,
  control = NULL,
  seed = NULL
) {
  weights_type <- match.arg(weights_type)

  if (!is.numeric(B) || length(B) != 1L || is.na(B) || B < 1) {
    stop("'B' must be a positive integer.", call. = FALSE)
  }
  B <- as.integer(B)

  if (!is.null(cluster) && length(cluster) != nrow(data)) {
    stop("'cluster' must have one entry per row of 'data'.", call. = FALSE)
  }

  control <- utils::modifyList(
    if (is.null(control)) list() else as.list(control),
    list(keep_tx = TRUE, return_hessian = TRUE)
  )

  fit <- fepoisson(
    formula = formula,
    data = data,
    weights = weights,
    offset = offset,
    control = control
  )

  if (is.null(fit[["tx"]]) || is.null(fit[["hessian"]]) || is.null(fit[["working_residuals"]]) || length(fit[["working_residuals"]]) != nrow(fit[["tx"]])) {
    stop("The fit does not contain the scores needed for the bootstrap.", call. = FALSE)
  }

  # Scores of the structural parameters; rows follow the observations kept in
  # the fit (after dropping NAs and separated observations)
  scores <- fit[["tx"]] * fit[["working_residuals"]]

  groups <- if (is.null(cluster)) {
    seq_len(nrow(scores))
  } else {
    cluster[fit[["model_indices"]]]
  }
  cluster_scores <- rowsum(scores, group = groups, reorder = FALSE)
  n_clusters <- nrow(cluster_scores)

  if (!is.null(seed)) {
    set.seed(seed)
  }

  w <- matrix(draw_wild_weights_(n_clusters * B, weights_type), n_clusters, B)

  # p x B perturbations of the estimator: H^{-1} sum_g W_g s_g
  delta <- solve(fit[["hessian"]], crossprod(cluster_scores, w))

  beta <- fit[["coef_table"]][, 1L]
  beta <- beta[!is.na(beta)]
  draws <- t(delta + beta)
  colnames(draws) <- names(beta)

  v <- stats::cov(draws)
  dimnames(v) <- list(names(beta), names(beta))

  q <- apply(draws, 2L, stats::quantile, probs = c(0.025, 0.975), names = FALSE)
  ci <- cbind(
    lower = 2 * beta - q[2L, ],
    upper = 2 * beta - q[1L, ]
  )
  rownames(ci) <- names(beta)

  structure(
    list(
      fit = fit,
      coefficients = beta,
      draws = draws,
      vcov = v,
      se = sqrt(diag(v)),
      ci = ci
    ),
    class = "wild_bootstrap_fepoisson"
  )
}

#' @title Wild bootstrap weights
#' @description Mean zero, unit variance weights
#' @param n number of weights
#' @param type weight distribution
#' @noRd
draw_wild_weights_ <- function(n, type) {
  switch(type,
    rademacher = sample(c(-1, 1), n, replace = TRUE),
    mammen = {
      # Mammen (1993) two-point distribution
      s5 <- sqrt(5)
      p <- (s5 + 1) / (2 * s5)
      ifelse(stats::runif(n) < p, -(s5 - 1) / 2, (s5 + 1) / 2)
    },
    normal = stats::rnorm(n)
  )
}

#' @export
print.wild_bootstrap_fepoisson <- function(x, ...) {
  cat("Wild score bootstrap (Kline and Santos, 2012):", nrow(x$draws), "draws\n\n")
  print(cbind(Estimate = x$coefficients, `Boot. SE` = x$se, x$ci), ...)
  invisible(x)
}
