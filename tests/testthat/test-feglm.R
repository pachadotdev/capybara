test_that("feglm is similar to glm", {
  # Gaussian ----

  # see felm

  # Poisson

  # see fepoisson

  # Binomial ----

  mod <- feglm(
    am ~ wt + mpg| cyl,
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
