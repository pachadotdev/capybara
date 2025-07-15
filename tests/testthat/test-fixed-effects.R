#' srr_stats (tests)
#' @srrstats {G5.6b} Conducts parameter recovery tests with multiple random seeds to validate consistency in results despite random components in data simulation or algorithms.
#' @srrstats {RE3.2} Compares fixed effects estimated by `feglm` and `felm` with equivalent GLM models to ensure similarity.
#' @srrstats {RE3.3} Validates the alignment of fixed effects recovery across different model implementations.
#' @srrstats {RE4.3} Ensures robustness of fixed effects recovery under varied random seeds.
#' @noRd
NULL

test_that("fixed_effects is similar to glm", {
  skip_on_cran()

  # Gaussian ----

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

  # TODO: numerical precision?
  expect_equal(rep(0, 10), c1 - c2, tolerance = 1e-1)

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

  expect_equal(rep(0, 10), c1 - c2, tolerance = 1e-2)

  # Binomial ----
  mod_binom <- feglm(am ~ wt + mpg | cyl, mtcars, family = binomial())
  mod_binom_base <- glm(am ~ wt + mpg + as.factor(cyl), mtcars, family = binomial())
  mod_binom_fixest <- fixest::feglm(am ~ wt + mpg | cyl, mtcars, family = binomial())

  mod_binom
  mod_binom_base
  mod_binom_fixest

  expect_equal(unname(coef(mod_binom) - coef(mod_binom_base)[2:3]), c(0, 0), tolerance = 1e-2)
  expect_equal(unname(coef(mod_binom) - coef(mod_binom_fixest)), c(0, 0), tolerance = 1e-2)

  fe_binom <- unname(drop(fixed_effects(mod_binom)$cyl))

  fe_binom_base <- coef(mod_binom_base)[c(1, 4, 5)]
  fe_binom_base <- unname(c(
    fe_binom_base[1],
    fe_binom_base[1] + fe_binom_base[2],
    fe_binom_base[1] + fe_binom_base[3]
  ))

  fe_binom_fixest <- fixest::fixef(mod_binom_fixest)

  fe_binom_base
  fe_binom_fixest$cyl

  expect_equal(fe_binom - fe_binom_base, c(0, 0, 0), tolerance = 1e-2)
  expect_equal(unname(fe_binom - fe_binom_fixest$cyl), c(0, 0, 0), tolerance = 1e-2)

  # Gamma ----
  mod_gamma <- feglm(mpg ~ wt + am | cyl, mtcars, family = Gamma())
  mod_gamma_base <- glm(mpg ~ wt + am + as.factor(cyl), mtcars, family = Gamma())

  expect_equal(unname(coef(mod_gamma) - coef(mod_gamma_base)[2:3]), c(0, 0), tolerance = 1e-2)

  fe_gamma <- unname(drop(fixed_effects(mod_gamma)$cyl))
  fe_gamma_base <- unname(coef(mod_gamma_base)[c(1, 4, 5)])
  fe_gamma_base <- c(
    fe_gamma_base[1],
    fe_gamma_base[1] + fe_gamma_base[2],
    fe_gamma_base[1] + fe_gamma_base[3]
  )
  
  expect_equal(fe_gamma - fe_gamma_base, c(0, 0, 0), tolerance = 1e-1)

  # Inverse Gaussian ----
  mod_invgauss <- feglm(mpg ~ wt + am | cyl, mtcars, family = inverse.gaussian())
  mod_invgauss_base <- glm(mpg ~ wt + am + as.factor(cyl), mtcars, family = inverse.gaussian())

  expect_equal(unname(coef(mod_invgauss) - coef(mod_invgauss_base)[2:3]), c(0, 0), tolerance = 1e-2)

  fe_invgauss <- unname(drop(fixed_effects(mod_invgauss)$cyl))
  fe_invgauss_base <- unname(coef(mod_invgauss_base)[c(1, 4, 5)])
  fe_invgauss_base <- c(
    fe_invgauss_base[1],
    fe_invgauss_base[1] + fe_invgauss_base[2],
    fe_invgauss_base[1] + fe_invgauss_base[3]
  )
  expect_equal(fe_invgauss - fe_invgauss_base, c(0, 0, 0), tolerance = 1e-2)
})
