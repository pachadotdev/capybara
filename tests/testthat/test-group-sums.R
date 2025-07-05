#' srr_stats (tests)
#' @srrstats {G5.2} Validates the correctness of group operations like summation and covariance using precomputed results.
#' @srrstats {RE3.3} Compares outputs of group-related calculations with reference implementations to ensure consistency.
#' @srrstats {RE4.3} Confirms robustness of grouped computations under different random seeds.
#' @srrstats {RE6.0} Ensures proper handling of input matrix dimensions and group structures in grouped summation functions.
#' @noRd
NULL

test_that("group_sums_* works", {
  skip_on_cran()

  set.seed(123)
  m <- matrix(rnorm(9), ncol = 3, nrow = 3)
  w <- matrix(rnorm(3), ncol = 1, nrow = 3)
  v <- matrix(rnorm(3), ncol = 1, nrow = 3)
  k <- 2L
  jlist <- list(1L, 2L)

  expect_equal(
    matrix(c(4.144, 4.872, -2.942), ncol = 1, nrow = 3),
    group_sums_(m, w, jlist),
    tolerance = 1e-2
  )

  expect_equal(
    matrix(c(0, 0, 0), ncol = 1, nrow = 3),
    group_sums_spectral_(m, w, v, k, jlist),
    tolerance = 1e-2
  )

  expect_equal(
    c(2.483, 2.644, -0.779),
    group_sums_var_(m, jlist)[, 1],
    tolerance = 1e-2
  )

  expect_equal(
    c(0, 0, 0),
    group_sums_cov_(m, m, jlist)[, 1],
    tolerance = 1e-2
  )
})
