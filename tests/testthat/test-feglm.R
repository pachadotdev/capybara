#' srr_stats (tests)
#' 
#' @srrstatsVerbose TRUE
#' 
#' @srrstats {G5.0} The tests use the widely known mtcars data set. It has few
#'  observations, and it is easy to compare the results with the base R
#'  functions.
#' @srrstats {G5.4b} We determine correctess for GLMs by comparison, checking
#'  the estimates versus base R and hardcoded values obtained with Alpaca
#'  (Stammann, 2018).
#' @srrstats {G5.8} **Edge condition tests** *to test that these conditions produce expected behaviour such as clear warnings or errors when confronted with data with extreme properties including but not limited to:*
#' @srrstats {G5.8b} *Data of unsupported types (e.g., character or complex numbers in for functions designed only for numeric data)*
#' @srrstats {RE7.2} Demonstrate that output objects retain aspects of input data such as row or case names (see **RE1.3**).
#' @srrstats {RE7.3} Demonstrate and test expected behaviour when objects returned from regression software are submitted to the accessor methods of **RE4.2**--**RE4.7**.
#' @srrstats {RE7.4} Extending directly from **RE4.15**, where appropriate, tests should demonstrate and confirm that forecast errors, confidence intervals, or equivalent values increase with forecast horizons.
#' 
#' @noRd
NULL

test_that("feglm is similar to glm", {
  # Gaussian ----

  # see test-felm.R

  # Poisson

  # see fepoisson

  # Binomial ----

  mod <- feglm(
    am ~ wt + mpg | cyl,
    mtcars,
    family = binomial()
  )

  mod_base <- glm(
    am ~ wt + mpg + as.factor(cyl),
    mtcars,
    family = binomial()
  )

  expect_equal(unname(round(coef(mod) - coef(mod_base)[2:3], 3)), rep(0, 2))

  fe <- unname(drop(fixed_effects(mod)$cyl))
  fe_base <- coef(mod_base)[c(1, 4, 5)]
  fe_base <- unname(fe_base + c(0, rep(fe_base[1], 2)))

  expect_equal(round(fe - fe_base, 2), rep(0, 3))

  # Gamma ----

  mod <- feglm(
    mpg ~ wt + am | cyl,
    mtcars,
    family = Gamma()
  )

  mod_base <- glm(
    mpg ~ wt + am + as.factor(cyl),
    mtcars,
    family = Gamma()
  )

  expect_equal(unname(round(coef(mod) - coef(mod_base)[2:3], 3)), rep(0, 2))

  fe <- unname(drop(fixed_effects(mod)$cyl))
  fe_base <- coef(mod_base)[c(1, 4, 5)]
  fe_base <- unname(fe_base + c(0, rep(fe_base[1], 2)))

  expect_equal(round(fe - fe_base, 2), rep(0, 3))

  # Inverse Gaussian ----

  mod <- feglm(
    mpg ~ wt + am | cyl,
    mtcars,
    family = inverse.gaussian()
  )

  mod_base <- glm(
    mpg ~ wt + am + as.factor(cyl),
    mtcars,
    family = inverse.gaussian()
  )

  expect_equal(unname(round(coef(mod) - coef(mod_base)[2:3], 3)), rep(0, 2))

  fe <- unname(drop(fixed_effects(mod)$cyl))
  fe_base <- coef(mod_base)[c(1, 4, 5)]
  fe_base <- unname(fe_base + c(0, rep(fe_base[1], 2)))

  expect_equal(round(fe - fe_base, 2), rep(0, 3))
})
