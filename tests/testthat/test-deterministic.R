#' srr_stats (tests)
#' @srrstats {G1.0} Implements tests to detect deterministic relations among predictors.
#' @srrstats {RE2.2} Ensures that models correctly fail when predictors are linearly dependent.
#' @srrstats {RE5.1} Confirms that the function provides meaningful error messages for invalid input.
#' @srrstats {RE5.2} Verifies that the model throws an error when dependent columns are included in the formula.
#' @srrstats {RE5.4} Checks robustness against deterministic linear relationships in the design matrix.
#' @srrstats {RE7.0} Exact relationships return a collinearity error.
#' @srrstats {RE7.0a} Perfectly noiseless input data is rejected, we have the `solve()` function for that.
#' @noRd
NULL

test_that("deterministic relations", {
  set.seed(123)
  d <- data.frame(
    y = rnorm(100),
    f = 1
  )

  d$x <- 2 * d$y
  d$x2 <- 2 * d$y

  # the solution is beta = 0.5 but we have the solve() function to
  # solve a linear system of equations!
  expect_error(coef(feglm(y ~ x | f, d)), "Linear dependent terms")

  # error because we check linear dependency in the data
  expect_error(feglm(y ~ x + x2 | f, d), "Linear dependent terms")
})
