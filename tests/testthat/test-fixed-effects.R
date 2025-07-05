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

  expect_equal(rep(0, 10), c1 - c2, tolerance = 1e-2)

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

  # TODO: check these numerical differences

  #  mod <- feglm(
  #    am ~ wt + mpg | cyl,
  #    mtcars,
  #    family = binomial()
  #  )

  #  mod_base <- glm(
  #    am ~ wt + mpg + as.factor(cyl),
  #    mtcars,
  #    family = binomial()
  #  )

  #  expect_equal(unname(coef(mod) - coef(mod_base)[2:3], 3), c(0, 0), tolerance = 1e-2)

  #  fe <- unname(drop(fixed_effects(mod)$cyl))
  #  fe_base <- coef(mod_base)[c(1, 4, 5)]
  #  fe_base <- unname(fe_base + c(0, rep(fe_base[1], 2)))

  #  expect_equal(fe - fe_base, c(0, 0, 0), tolerance = 1e-2)

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

  expect_equal(unname(coef(mod) - coef(mod_base)[2:3]), c(0, 0), tolerance = 1e-2)

  # TODO: check these numerical differences

  #  fe <- unname(drop(fixed_effects(mod)$cyl))
  #  fe_base <- coef(mod_base)[c(1, 4, 5)]
  #  fe_base <- unname(fe_base + c(0, rep(fe_base[1], 2)))

  #  expect_equal(fe - fe_base, c(0, 0, 0), tolerance = 1e-2)

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

  expect_equal(unname(coef(mod) - coef(mod_base)[2:3]), c(0, 0), tolerance = 1e-2)

  fe <- unname(drop(fixed_effects(mod)$cyl))
  fe_base <- coef(mod_base)[c(1, 4, 5)]
  fe_base <- unname(fe_base + c(0, rep(fe_base[1], 2)))

  expect_equal(fe - fe_base, c(0, 0, 0), tolerance = 1e-2)
})
