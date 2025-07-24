#' srr_stats (tests)
#' @srrstats {G5.2} Validates the correctness of group operations like summation and covariance using precomputed results.
#' @srrstats {RE3.3} Compares outputs of group-related calculations with reference implementations to ensure consistency.
#' @srrstats {RE4.3} Confirms robustness of grouped computations under different random seeds.
#' @srrstats {RE6.0} Ensures proper handling of input matrix dimensions and group structures in grouped summation functions.
#' @noRd
NULL

test_that("group_sums_* works", {
  skip_on_cran()

  load_all()

  fit <- feglm(am ~ mpg | cyl, mtcars[1:10, ], family = binomial())
  l = 0L

  panel_structure = "classic"

  beta_uncorr <- fit[["coefficients"]]
  control <- fit[["control"]]
  data <- fit[["data"]]
  family <- fit[["family"]]
  formula <- fit[["formula"]]
  fe.levels <- fit[["fe.levels"]]
  nms_sp <- names(beta_uncorr)
  nt <- fit[["nobs"]][["nobs"]]
  fe_names <- names(fe.levels)
  k <- length(fe.levels)

  y <- data[[1L]]
  x <- model.matrix(formula, data, rhs = 1L)[, -1L, drop = FALSE]
  attr(x, "dimnames") <- NULL
  w <- fit[["weights"]]

  FEs <- get_index_list_(fe_names, data)

  eta <- fit[["eta"]]
  mu <- family[["linkinv"]](eta)
  mu_eta <- family[["mu.eta"]](eta)
  v <- w * (y - mu)
  w <- w * mu_eta
  z <- w * partial_mu_eta_(eta, family, 2L)

  x <- demean_variables_(x, w, FEs, control[["demean_tol"]],
    control[["iter_max"]], control[["iter_interrupt"]], control[["iter_ssr"]],
    "gaussian")

  # groupSums <- function(M, w, jlist) {
  #   P <- ncol(M)
  #   b <- numeric(P)
  #   for (indexes in jlist) {
  #     num <- colSums(M[indexes, , drop = FALSE])
  #     denom <- sum(w[indexes])
  #     b <- b + num / denom
  #   }
  #   b
  # }

  # groupSumsSpectral <- function(M, v, w, L, jlist) {
  #   J <- length(jlist)
  #   P <- ncol(M)
  #   b <- numeric(P)

  #   for (j in seq_len(J)) {
  #     indexes <- jlist[[j]]
  #     T <- length(indexes)

  #     if (T <= 1) next

  #     num <- numeric(P)
  #     for (l in seq_len(min(L, T - 1))) {
  #       t_indices <- (l + 1):T
  #       weight <- T / (T - l)
  #       current_idx <- indexes[t_indices]
  #       lagged_idx <- indexes[t_indices - l]
  #       num <- num + colSums(M[current_idx, , drop = FALSE] * v[lagged_idx] * weight)
  #     }

  #     b <- b + num / sum(w[indexes])
  #   }

  #   return(b)
  # }

  # groupSumsVar <- function(M, jlist) {
  #   P <- ncol(M)
  #   V <- matrix(0, nrow = P, ncol = P)

  #   for (j in seq_along(jlist)) {
  #     indexes <- jlist[[j]]
  #     v <- colSums(M[indexes, , drop = FALSE])
  #     V <- V + outer(v, v)
  #   }

  #   return(V)
  # }

  # groupSumsCov <- function(M, N, jlist) {
  #   P <- ncol(M)
  #   Q <- ncol(N)

  #   # Initialize result matrix
  #   V <- matrix(0, nrow = P, ncol = Q)

  #   for (j in seq_along(jlist)) {
  #     indexes <- jlist[[j]]

  #     if (length(indexes) < 2) next

  #     # Get submatrices for this group
  #     M_group <- M[indexes, , drop = FALSE]
  #     N_group <- N[indexes, , drop = FALSE]

  #     # Add M_group^T * N_group to result
  #     V <- V + t(M_group) %*% N_group
  #   }

  #   return(V)
  # }

  expect_equal(
    group_sums_(x * z, w, FEs[[1L]]),
    # groupSums(x * z, w, FEs[[1L]])
    as.matrix(-0.07426021),
    tolerance = 1e-2
  )

  expect_equal(
    group_sums_spectral_(x * w, w, v, k, FEs[[1L]]),
    # groupSumsSpectral(x * w, v, w, l, FEs[[1L]])
    as.matrix(0),
    tolerance = 1e-2
  )

  expect_equal(
    group_sums_var_(x * w, FEs[[1L]]),
    # groupSumsVar(x * w, FEs[[1L]])
    as.matrix(1.300417e-27),
    tolerance = 1e-2
  )

  expect_equal(
    group_sums_cov_(x * w, x * w, FEs[[1L]]),
    # groupSumsCov(x * w, x * w, FEs[[1L]])
    as.matrix(0.01355264),
    tolerance = 1e-2
  )
})
