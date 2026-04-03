#' srr_stats (tests)
#' @srrstats {G1.0} Implements unit testing for APEs and bias correction using known examples.
#' @srrstats {G2.1} Validates the correct implementation of model outputs (e.g., coefficients, summaries).
#' @srrstats {G2.2} Compares results against known benchmarks from alternative implementations.
#' @srrstats {G3.0} Ensures that printed outputs for models are as expected.
#' @srrstats {RE2.1} Verifies that computed APEs align with external library outputs (`alpaca`).
#' @srrstats {RE2.3} Confirms that bias correction results are consistent with benchmark values.
#' @srrstats {RE5.2} Ensures coefficients, summaries, and bias corrections are computed without errors.
#' @srrstats {RE6.1} Ensures efficient computation for moderately sized data (e.g., subsetted trade data).
#' @noRd
NULL

# ---- APES tests (C++ implementation via fit_control) ----

test_that("compute_apes works via fit_control", {
  skip_on_cran()

  trade_short <- yotov2017[yotov2017$exp_year == "CAN1994", ]
  trade_short <- trade_short[trade_short$trade > 100, ]
  trade_short$trade <- ifelse(trade_short$trade > 200, 1L, 0L)

  mod <- feglm(trade ~ lang | exp_year, trade_short, family = binomial(),
               control = fit_control(compute_apes = TRUE))

  expect_s3_class(mod, "feglm")
  expect_true(isTRUE(mod$has_apes))
  expect_true(length(mod$apes_delta) == 1)
  expect_true(is.matrix(mod$apes_vcov))

  # the values come from:
  # mod2 <- alpaca::feglm(trade ~ lang | year, trade_short, family = binomial())
  # apes2 <- alpaca::getAPEs(mod2)
  apes2 <- c("lang" = 0.05)

  expect_equal(mod$apes_delta, apes2, tolerance = 1e-1)
})

test_that("compute_apes works with binary response", {
  # Simulate larger dataset to avoid separation issues
  set.seed(123)
  n <- 100
  sim_data <- data.frame(
    y = rbinom(n, 1, 0.5),
    x = rnorm(n),
    fe = sample(1:5, n, replace = TRUE)
  )

  mod <- feglm(y ~ x | fe, sim_data, family = binomial(),
               control = fit_control(compute_apes = TRUE))

  expect_s3_class(mod, "feglm")
  expect_true(isTRUE(mod$has_apes))
  expect_true(length(mod$apes_delta) == 1)
  expect_true(names(mod$apes_delta) == "x")
})

test_that("apes_delta has names", {
  # Simulate larger dataset to avoid separation issues
  set.seed(123)
  n <- 100
  sim_data <- data.frame(
    y = rbinom(n, 1, 0.5),
    x = rnorm(n),
    fe = sample(1:5, n, replace = TRUE)
  )

  mod <- feglm(y ~ x | fe, sim_data, family = binomial(),
               control = fit_control(compute_apes = TRUE))

  expect_true(!is.null(names(mod$apes_delta)))
  expect_equal(names(mod$apes_delta), "x")
})

test_that("apes_vcov has dimnames", {
  # Simulate larger dataset to avoid separation issues
  set.seed(123)
  n <- 100
  sim_data <- data.frame(
    y = rbinom(n, 1, 0.5),
    x = rnorm(n),
    fe = sample(1:5, n, replace = TRUE)
  )

  mod <- feglm(y ~ x | fe, sim_data, family = binomial(),
               control = fit_control(compute_apes = TRUE))

  expect_true(!is.null(dimnames(mod$apes_vcov)))
  expect_equal(rownames(mod$apes_vcov), "x")
  expect_equal(colnames(mod$apes_vcov), "x")
})

# ---- Bias correction tests (C++ implementation via fit_control) ----

test_that("compute_bias_corr works via fit_control", {
  skip_on_cran()

  trade_short <- yotov2017[yotov2017$exp_year == "CAN1994", ]
  trade_short <- trade_short[trade_short$trade > 100, ]
  trade_short$trade <- ifelse(trade_short$trade > 200, 1L, 0L)

  mod <- feglm(trade ~ lang | exp_year, trade_short, family = binomial(),
               control = fit_control(compute_bias_corr = TRUE))

  expect_s3_class(mod, "feglm")
  expect_true(isTRUE(mod$has_bias_corr))
  expect_true(length(mod$beta_uncorrected) == 1)
  expect_true(length(mod$bias_corr_term) == 1)

  # the values come from:
  # mod2 <- alpaca::feglm(trade ~ lang | year, trade_short, family = binomial())
  # bias2 <- alpaca::biasCorr(mod2)
  bias2 <- c("lang" = 0.2436)

  # The corrected coefficient should be in coef_table
  expect_equal(trunc(coef(mod), 2), trunc(bias2, 2))
})

test_that("compute_bias_corr works with binary response", {
  # Simulate larger dataset to avoid separation issues
  set.seed(123)
  n <- 100
  sim_data <- data.frame(
    y = rbinom(n, 1, 0.5),
    x = rnorm(n),
    fe = sample(1:5, n, replace = TRUE)
  )

  mod <- feglm(y ~ x | fe, sim_data, family = binomial(),
               control = fit_control(compute_bias_corr = TRUE))

  expect_s3_class(mod, "feglm")
  expect_true(isTRUE(mod$has_bias_corr))
  expect_true(length(coef(mod)) == 1)
  expect_true(length(mod$beta_uncorrected) == 1)
})

test_that("bias_corr preserves uncorrected coefficients", {
  # Simulate larger dataset to avoid separation issues
  set.seed(123)
  n <- 100
  sim_data <- data.frame(
    y = rbinom(n, 1, 0.5),
    x = rnorm(n),
    fe = sample(1:5, n, replace = TRUE)
  )

  mod <- feglm(y ~ x | fe, sim_data, family = binomial(),
               control = fit_control(compute_bias_corr = TRUE))

  # beta_uncorrected should be stored
  expect_true(!is.null(mod$beta_uncorrected))
  expect_true(names(mod$beta_uncorrected) == "x")
})

# ---- Combined APES and bias correction ----

test_that("compute_apes and compute_bias_corr work together", {
  # Simulate larger dataset to avoid separation issues
  set.seed(123)
  n <- 100
  sim_data <- data.frame(
    y = rbinom(n, 1, 0.5),
    x = rnorm(n),
    fe = sample(1:5, n, replace = TRUE)
  )

  mod <- feglm(y ~ x | fe, sim_data, family = binomial(),
               control = fit_control(compute_apes = TRUE, compute_bias_corr = TRUE))

  expect_s3_class(mod, "feglm")
  expect_true(isTRUE(mod$has_apes))
  expect_true(isTRUE(mod$has_bias_corr))
  expect_true(length(mod$apes_delta) == 1)
  expect_true(length(mod$beta_uncorrected) == 1)
})

# ---- APES/bias_corr not computed for non-binomial ----

test_that("compute_apes is ignored for non-binary models", {
  mod <- fepoisson(mpg ~ wt | cyl, mtcars,
                   control = fit_control(compute_apes = TRUE))

  expect_false(isTRUE(mod$has_apes))
})

test_that("compute_bias_corr is ignored for non-binary models", {
  mod <- fepoisson(mpg ~ wt | cyl, mtcars,
                   control = fit_control(compute_bias_corr = TRUE))

  expect_false(isTRUE(mod$has_bias_corr))
})

# ---- Stammann centering ----

test_that("compute_apes works with stammann centering", {
  skip_on_cran()
  ctrl <- fit_control(centering = "stammann", compute_apes = TRUE)

  trade_short <- yotov2017[yotov2017$exp_year == "CAN1994", ]
  trade_short <- trade_short[trade_short$trade > 100, ]
  trade_short$trade <- ifelse(trade_short$trade > 200, 1L, 0L)

  mod <- feglm(trade ~ lang | exp_year, trade_short, family = binomial(), control = ctrl)

  expect_s3_class(mod, "feglm")
  expect_true(isTRUE(mod$has_apes))
  
  apes2 <- c("lang" = 0.05)
  expect_equal(mod$apes_delta, apes2, tolerance = 1e-1)
})

test_that("compute_apes with stammann centering", {
  ctrl <- fit_control(centering = "stammann", compute_apes = TRUE)
  # Simulate larger dataset to avoid separation issues
  set.seed(123)
  n <- 100
  sim_data <- data.frame(
    y = rbinom(n, 1, 0.5),
    x = rnorm(n),
    fe = sample(1:5, n, replace = TRUE)
  )

  mod <- feglm(y ~ x | fe, sim_data, family = binomial(), control = ctrl)

  expect_s3_class(mod, "feglm")
  expect_true(isTRUE(mod$has_apes))
  expect_true(length(mod$apes_delta) == 1)
})

test_that("compute_bias_corr works with mtcars (stammann centering)", {
  ctrl <- fit_control(centering = "stammann", compute_bias_corr = TRUE)
  # Simulate larger dataset to avoid separation issues
  set.seed(123)
  n <- 100
  sim_data <- data.frame(
    y = rbinom(n, 1, 0.5),
    x = rnorm(n),
    fe = sample(1:5, n, replace = TRUE)
  )

  mod <- feglm(y ~ x | fe, sim_data, family = binomial(), control = ctrl)

  expect_s3_class(mod, "feglm")
  expect_true(isTRUE(mod$has_bias_corr))
  expect_true(length(coef(mod)) == 1)
})

test_that("compute_apes and compute_bias_corr work together (stammann centering)", {
  ctrl <- fit_control(centering = "stammann", compute_apes = TRUE, compute_bias_corr = TRUE)
  # Simulate larger dataset to avoid separation issues
  set.seed(123)
  n <- 100
  sim_data <- data.frame(
    y = rbinom(n, 1, 0.5),
    x = rnorm(n),
    fe = sample(1:5, n, replace = TRUE)
  )

  mod <- feglm(y ~ x | fe, sim_data, family = binomial(), control = ctrl)

  expect_s3_class(mod, "feglm")
  expect_true(isTRUE(mod$has_apes))
  expect_true(isTRUE(mod$has_bias_corr))
})
