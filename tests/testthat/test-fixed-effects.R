#' srr_stats (tests)
#'
#' @srrstatsVerbose TRUE
#' 
#' @srrstats {G5.6} **Parameter recovery tests** *to test that the implementation produce expected results given data with known properties. For instance, a linear regression algorithm should return expected coefficient values for a simulated data set generated from a linear model.*
#' @srrstats {G5.6a} *Parameter recovery tests should generally be expected to succeed within a defined tolerance rather than recovering exact values.*
#' @srrstats {G5.6b} *Parameter recovery tests should be run with multiple random seeds when either data simulation or the algorithm contains a random component. (When long-running, such tests may be part of an extended, rather than regular, test suite; see G5.10-4.12, below).*
#' 
#' @noRd
NULL

test_that("fixed_effects is similar to glm", {
  # this is also tested in feglm with real data
  # this is about  SRR {G5.6b}:
  # Parameter recovery tests should be run with multiple random seeds when
  # either data simulation or the algorithm contains a random component.

  set.seed(200100)
  d <- data.frame(
    y = rnorm(100),
    x = rnorm(100),
    f = factor(sample(1:10, 1000, replace = TRUE))
  )

  fit1 <- glm(y ~ x + f + 0, data = d)
  fit2 <- feglm(y ~ x | f, data = d, family = gaussian())

  c1 <- unname(coef(fit1)[grep("f", names(coef(fit1)))])
  c2 <- unname(drop(fixed_effects(fit2)$f))

  expect_equal(round(c1 - c2, 3), rep(0, 10))

  set.seed(100200)
  d <- data.frame(
    y = rnorm(100),
    x = rnorm(100),
    f = factor(sample(1:10, 1000, replace = TRUE))
  )

  fit1 <- lm(y ~ x + f + 0, data = d)
  fit2 <- felm(y ~ x | f, data = d)

  c1 <- unname(coef(fit1)[grep("f", names(coef(fit1)))])
  c2 <- unname(drop(fixed_effects(fit2)$f))

  expect_equal(round(c1 - c2, 3), rep(0, 10))
})
