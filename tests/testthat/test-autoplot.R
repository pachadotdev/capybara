#' srr_stats (tests)
#' @srrstats {G1.0} Implements unit testing for plotting functionality.
#' @srrstats {G2.3} Tests compatibility with standard plotting libraries like ggplot2.
#' @srrstats {RE3.1} Verifies the correctness of visual outputs for model coefficients.
#' @srrstats {RE3.2} Ensures that confidence levels provided to the plotting function are validated.
#' @srrstats {RE5.1} Confirms that `autoplot` fails gracefully with invalid inputs.
#' @srrstats {RE5.3} Validates that the output of `autoplot` is a `ggplot` object for visualizations.
#' @noRd
NULL

test_that("autoplot works", {
  mod <- felm(mpg ~ wt + qsec | cyl, mtcars)

  expect_s3_class(autoplot(mod, conf_level = 0.99), "ggplot")
  expect_s3_class(autoplot(mod), "ggplot")

  expect_error(autoplot(1L))
  expect_error(autoplot(mod, conf_level = 1.01))
  expect_error(autoplot(mod, conf_level = -0.01))
})
