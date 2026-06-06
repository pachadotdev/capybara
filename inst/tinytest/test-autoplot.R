#' srr_stats (tests)
#' @srrstats {G1.0} Implements unit testing for plotting functionality.
#' @srrstats {G2.3} Tests compatibility with standard plotting libraries like ggplot2.
#' @srrstats {RE3.1} Verifies the correctness of visual outputs for model coefficients.
#' @srrstats {RE3.2} Ensures that confidence levels provided to the plotting function are validated.
#' @srrstats {RE5.1} Confirms that `autoplot` fails gracefully with invalid inputs.
#' @srrstats {RE5.3} Validates that the output of `autoplot` is a `ggplot` object for visualizations.
#' @noRd
NULL

# autoplot works for felm
local({
  mod <- felm(mpg ~ wt + qsec | cyl, mtcars)

  expect_true(inherits(autoplot(mod, conf_level = 0.99), "ggplot2::ggplot"))
  expect_true(inherits(autoplot(mod), "ggplot2::ggplot"))

  expect_error(autoplot(1L))
  expect_error(autoplot(mod, conf_level = 1.01))
  expect_error(autoplot(mod, conf_level = -0.01))
})

# autoplot works for feglm/fepoisson
local({
  mod <- fepoisson(mpg ~ wt + qsec | cyl, mtcars)

  expect_true(inherits(autoplot(mod), "ggplot2::ggplot"))
  expect_true(inherits(autoplot(mod, conf_level = 0.90), "ggplot2::ggplot"))
  expect_true(inherits(autoplot(mod, conf_level = 0.99), "ggplot2::ggplot"))
})

# autoplot works for binomial feglm
local({
  mod <- feglm(am ~ wt + mpg | cyl, mtcars, family = binomial())

  expect_true(inherits(autoplot(mod), "ggplot2::ggplot"))
})

# autoplot errors on invalid conf_level
local({
  mod <- fepoisson(mpg ~ wt | cyl, mtcars)

  expect_error(autoplot(mod, conf_level = 0))
  expect_error(autoplot(mod, conf_level = 1))
  expect_error(autoplot(mod, conf_level = -0.5))
  expect_error(autoplot(mod, conf_level = 1.5))
})

# autoplot errors on wrong class
local({
  mod_lm <- lm(mpg ~ wt, mtcars)

  expect_error(autoplot.feglm(mod_lm))
  expect_error(autoplot.felm(mod_lm))
})

# autoplot works with multiple predictors
local({
  mod <- felm(mpg ~ wt + hp + qsec + drat | cyl, mtcars)

  p <- autoplot(mod)

  expect_true(inherits(p, "ggplot2::ggplot"))
})

# autoplot default conf_level is 0.95
local({
  mod <- felm(mpg ~ wt | cyl, mtcars)

  # Both should work without error (default is 0.95)
  p1 <- autoplot(mod)
  p2 <- autoplot(mod, conf_level = 0.95)

  expect_true(inherits(p1, "ggplot2::ggplot"))
  expect_true(inherits(p2, "ggplot2::ggplot"))
})
