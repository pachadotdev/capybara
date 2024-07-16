test_that("group_sums_* works", {
  set.seed(123)
  M <- matrix(rnorm(9), ncol = 3, nrow = 3)
  w <- matrix(rnorm(3), ncol = 1, nrow = 3)
  v <- matrix(rnorm(3), ncol = 1, nrow = 3)
  K <- 2L
  jlist <- list(1L, 2L)

  expect_equal(
    round(group_sums_(M, w, jlist), 3),
    # round(alpaca:::groupSums(M, w, jlist), 3)
    matrix(c(4.144, 4.872, -2.942), ncol = 1, nrow = 3)
  )

  expect_equal(
    round(group_sums_spectral_(M, w, v, K, jlist), 3),
    # round(alpaca:::groupSumsSpectral(M, w, v, K, jlist), 3)
    matrix(c(0, 0, 0), ncol = 1, nrow = 3)
  )

  expect_equal(
    round(group_sums_var_(M, jlist)[, 1], 3),
    # round(alpaca:::groupSumsVar(M, jlist)[, 1], 3)
    c(2.483, 2.644, -0.779)
  )

  expect_equal(
    round(group_sums_cov_(M, M, jlist)[, 1], 3),
    # round(alpaca:::groupSumsCov(M, M, jlist)[, 1], 3)
    c(0, 0, 0)
  )
})
