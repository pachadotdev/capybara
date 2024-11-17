#' srr_stats (tests)
#' @srrstatsVerbose TRUE
#' @srrstats {RE7.0} Test exact relationships between predictors.
#' @srrstats {RE7.0a} Reject perfectly noiseless input data.
#' @srrstats {RE7.1} Tests exact relationships between predictor and response.
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
  expect_error(coef(feglm(y ~ x |f, d)), "Linear dependent terms")

  # error because we check linear dependency in the data
  expect_error(feglm(y ~ x + x2 | f, d), "Linear dependent terms")
})
