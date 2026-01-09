#' srr_stats (tests)
#' @srrstats {G1.0} Implements unit testing for plotting functionality.
#' @srrstats {G2.3} Tests compatibility with standard plotting libraries like ggplot2.
#' @srrstats {RE3.1} Verifies the correctness of visual outputs for model coefficients.
#' @srrstats {RE3.2} Ensures that confidence levels provided to the plotting function are validated.
#' @srrstats {RE5.1} Confirms that `autoplot` fails gracefully with invalid inputs.
#' @srrstats {RE5.3} Validates that the output of `autoplot` is a `ggplot` object for visualizations.
#' @noRd
NULL

test_that("autoplot works for felm", {
  mod <- felm(mpg ~ wt + qsec | cyl, mtcars)

  expect_s3_class(autoplot(mod, conf_level = 0.99), "ggplot")
  expect_s3_class(autoplot(mod), "ggplot")

  expect_error(autoplot(1L))
  expect_error(autoplot(mod, conf_level = 1.01))
  expect_error(autoplot(mod, conf_level = -0.01))
})

test_that("autoplot works for feglm/fepoisson", {
  mod <- fepoisson(mpg ~ wt + qsec | cyl, mtcars)

  expect_s3_class(autoplot(mod), "ggplot")
  expect_s3_class(autoplot(mod, conf_level = 0.90), "ggplot")
  expect_s3_class(autoplot(mod, conf_level = 0.99), "ggplot")
})

test_that("autoplot works for binomial feglm", {
  mod <- feglm(am ~ wt + mpg | cyl, mtcars, family = binomial())

  expect_s3_class(autoplot(mod), "ggplot")
})

test_that("autoplot errors on invalid conf_level", {
  mod <- fepoisson(mpg ~ wt | cyl, mtcars)

  expect_error(autoplot(mod, conf_level = 0))
  expect_error(autoplot(mod, conf_level = 1))
  expect_error(autoplot(mod, conf_level = -0.5))
  expect_error(autoplot(mod, conf_level = 1.5))
})

test_that("autoplot errors on wrong class", {
  mod_lm <- lm(mpg ~ wt, mtcars)

  expect_error(autoplot.feglm(mod_lm))
  expect_error(autoplot.felm(mod_lm))
})

test_that("autoplot works with multiple predictors", {
  mod <- felm(mpg ~ wt + hp + qsec + drat | cyl, mtcars)

  p <- autoplot(mod)

  expect_s3_class(p, "ggplot")
})

test_that("autoplot default conf_level is 0.95", {
  mod <- felm(mpg ~ wt | cyl, mtcars)

  # Both should work without error (default is 0.95)
  p1 <- autoplot(mod)
  p2 <- autoplot(mod, conf_level = 0.95)

  expect_s3_class(p1, "ggplot")
  expect_s3_class(p2, "ggplot")
})
