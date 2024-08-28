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
