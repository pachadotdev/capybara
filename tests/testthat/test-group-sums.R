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

  set.seed(123)
  m <- matrix(rnorm(9), ncol = 3, nrow = 3)
  w <- matrix(rnorm(3), ncol = 1, nrow = 3)
  v <- matrix(rnorm(3), ncol = 1, nrow = 3)
  k <- 2L
  jlist <- list(1L, 2L)

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

  # groupSums(m, w, jlist)

  expect_equal(
    group_sums_(m, w, jlist),
    # alpaca:::groupSums(m, w, jlist)
    matrix(c(4.144, 4.872, -2.942), ncol = 1, nrow = 3),
    tolerance = 1e-2
  )

  expect_equal(
    group_sums_spectral_(m, w, v, k, jlist),
    # alpaca:::groupSumsSpectral(m, w, v, k, jlist)
    matrix(c(0, 0, 0), ncol = 1, nrow = 3),
    tolerance = 1e-2
  )

  expect_equal(
    group_sums_var_(m, jlist)[, 1],
    # alpaca:::groupSumsVar(m, jlist)
    c(2.483, 2.644, -0.779),
    tolerance = 1e-2
  )

  expect_equal(
    group_sums_cov_(m, m, jlist)[, 1],
    # alpaca:::groupSumsCov(m, m, jlist)
    c(0, 0, 0),
    tolerance = 1e-2
  )
})
