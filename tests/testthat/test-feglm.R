#' srr_stats (tests)
#' @srrstats {RE2.1} Ensures that models throw meaningful error messages when input parameters or data are invalid.
#' @srrstats {RE3.2} Compares model outputs (coefficients and fixed effects) against established benchmarks like base R's `glm`.
#' @srrstats {RE3.3} Confirms consistency of fixed effects and structural parameters between `feglm` and equivalent base models.
#' @srrstats {RE5.1} Validates appropriate error handling for omitted arguments, such as missing formula or data.
#' @srrstats {RE5.2} Confirms that incorrect control settings result in appropriate error messages.
#' @srrstats {RE5.3} Verifies that the function stops execution when given unsupported model families or inappropriate responses.
#' @srrstats {RE5.4} Ensures that the model gracefully handles invalid starting values for beta, eta, or theta.
#' @srrstats {RE6.0} Implements robust testing for invalid combinations of fixed effects or missing parameters in APEs and GLMs.
#' @srrstats {RE7.1} Validates consistency in output types and structures across all supported families and link functions.
#' @srrstats {RE7.2} Confirms that confidence intervals and standard errors are computed correctly for coefficients.
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
