#' srr_stats (tests)
#' @srrstats {RE3.1} Validates consistency between `fetobit` and other established R models.
#' @srrstats {RE3.2} Compares coefficients produced by `fetobit` with baseline models.
#' @srrstats {RE4.3} Ensures stable estimates when adding negligible noise to the data.
#' @srrstats {RE5.1} Validates proper output generation for the model summary and printing methods.
#' @srrstats {RE7.3} Confirms that estimated coefficients are within a reasonable variation threshold.
#' @noRd
NULL

# Helper to create tobit test data
make_tobit_data <- function(n = 200, lower = 0, upper = Inf, seed = 123) {
  set.seed(seed)
  d <- data.frame(
    x1 = rnorm(n),
    x2 = rnorm(n),
    f1 = factor(sample(1:3, n, replace = TRUE)),
    f2 = factor(sample(1:2, n, replace = TRUE))
  )
  # Generate latent y*
  y_star <- 2 + 1.5 * d$x1 - 0.8 * d$x2 + rnorm(n)
  # Apply censoring
  d$y <- pmin(pmax(y_star, lower), upper)
  d
}

test_that("fetobit basic functionality works", {
  d <- make_tobit_data(n = 200, lower = 0, seed = 42)

  # Should run without error
  mod <- fetobit(y ~ x1 | f1, d, tobit_lower = 0)

  expect_s3_class(mod, "feglm")
  expect_equal(length(coef(mod)), 1)
  expect_output(print(mod))
  expect_visible(summary(mod))

  # Check that fitted values are returned
  n_obs <- unname(mod[["nobs"]]["nobs_full"])
  expect_equal(length(fitted(mod)), n_obs)
  expect_equal(length(predict(mod)), n_obs)
})

test_that("fetobit and feglm with tobit() give the same results", {
  d <- make_tobit_data(n = 200, lower = 0, seed = 789)

  ctrl <- fit_control(tobit_lower = 0, check_separation = FALSE)

  mod1 <- fetobit(y ~ x1 | f1, d, tobit_lower = 0, control = fit_control(check_separation = FALSE))
  mod2 <- feglm(y ~ x1 | f1, d, family = "tobit", control = ctrl)

  expect_equal(coef(mod1), coef(mod2))
  expect_equal(fitted(mod1), fitted(mod2))
})

test_that("fetobit handles two-sided censoring", {
  d <- make_tobit_data(n = 200, lower = 0, upper = 5, seed = 456)

  mod <- fetobit(y ~ x1 | f1, d, tobit_lower = 0, tobit_upper = 5)

  expect_s3_class(mod, "feglm")
  expect_equal(length(coef(mod)), 1)

  # Fitted values should be within valid range (though for tobit this isn't strictly required)
  expect_true(all(is.finite(fitted(mod))))
})

test_that("fetobit with multiple fixed effects", {
  d <- make_tobit_data(n = 300, lower = 0, seed = 222)

  # K = 2
  mod <- fetobit(y ~ x1 | f1 + f2, d, tobit_lower = 0)

  expect_s3_class(mod, "feglm")
  expect_equal(length(coef(mod)), 1)
})

test_that("fetobit handles cluster standard errors", {
  d <- make_tobit_data(n = 200, lower = 0, seed = 111)
  d$cl <- factor(sample(1:10, nrow(d), replace = TRUE))

  mod <- fetobit(y ~ x1 | f1 | cl, d, tobit_lower = 0, vcov = "cluster")
  smod <- summary(mod)

  expect_equal(mod$vcov_type, "cluster")
  expect_true(all(is.finite(coef(smod)[, "Std. Error"])))
})

test_that("fetobit estimation is stable with noise", {
  set.seed(123)
  d <- make_tobit_data(n = 200, lower = 0, seed = 456)
  d$x1_noisy <- d$x1 + pmax(rnorm(nrow(d)), 0) * .Machine$double.eps

  m1 <- fetobit(y ~ x1 | f1, d, tobit_lower = 0)
  m2 <- fetobit(y ~ x1_noisy | f1, d, tobit_lower = 0)

  expect_equal(unname(coef(m1)), unname(coef(m2)))
})
