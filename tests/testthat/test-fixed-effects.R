#' srr_stats (tests)
#' @srrstats {G5.6b} Conducts parameter recovery tests with multiple random seeds to validate consistency in results despite random components in data simulation or algorithms.
#' @srrstats {RE3.2} Compares fixed effects estimated by `feglm` and `felm` with equivalent GLM models to ensure similarity.
#' @srrstats {RE3.3} Validates the alignment of fixed effects recovery across different model implementations.
#' @srrstats {RE4.3} Ensures robustness of fixed effects recovery under varied random seeds.
#' @noRd
NULL

test_that("fixed_effects is similar to glm", {
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
