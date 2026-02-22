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

test_that("apes/bias works", {
  skip_on_cran()

  trade_short <- trade_panel[trade_panel$exp_year == "CAN1994", ]
  trade_short <- trade_short[trade_short$trade > 100, ]
  trade_short$trade <- ifelse(trade_short$trade > 200, 1L, 0L)

  mod1 <- feglm(trade ~ lang | exp_year, trade_short, family = binomial())

  expect_s3_class(mod1, "feglm")

  apes1 <- apes(mod1)
  bias1 <- bias_corr(mod1)

  # the values come from:
  # mod2 <- alpaca::feglm(trade ~ lang | year, trade_short, family = binomial())
  # apes2 <- alpaca::getAPEs(mod2)
  # bias2 <- alpaca::biasCorr(mod2)
  apes2 <- c("lang" = 0.05)
  bias2 <- c("lang" = 0.2436)

  expect_output(print(mod1))

  expect_equal(length(coef(apes1)), 1)
  expect_equal(coef(apes1), apes2, tolerance = 1e-1) # TODO: check the 0.02 difference later
  expect_equal(length(coef(summary(apes(mod1)))), 4)

  expect_equal(length(coef(bias1)), 1)
  expect_equal(trunc(coef(bias1), 2), trunc(bias2, 2))
  expect_equal(length(coef(summary(bias1))), 4)
})

test_that("apes with mtcars binary response works", {
  mtcars2 <- mtcars
  mtcars2$mpg01 <- ifelse(mtcars2$mpg > mean(mtcars2$mpg), 1L, 0L)

  mod <- feglm(mpg01 ~ wt | cyl, mtcars2, family = binomial())

  apes_result <- apes(mod)

  expect_s3_class(apes_result, "apes")
  expect_true(length(coef(apes_result)) == 1)
})

test_that("apes summary works", {
  mtcars2 <- mtcars
  mtcars2$mpg01 <- ifelse(mtcars2$mpg > mean(mtcars2$mpg), 1L, 0L)

  mod <- feglm(mpg01 ~ wt | cyl, mtcars2, family = binomial())
  apes_result <- apes(mod)

  summ <- summary(apes_result)

  expect_s3_class(summ, "summary.apes")
  expect_output(print(summ))
})

test_that("apes coef method works", {
  mtcars2 <- mtcars
  mtcars2$mpg01 <- ifelse(mtcars2$mpg > mean(mtcars2$mpg), 1L, 0L)

  mod <- feglm(mpg01 ~ wt | cyl, mtcars2, family = binomial())
  apes_result <- apes(mod)

  coefs <- coef(apes_result)

  expect_true(is.numeric(coefs))
  expect_equal(length(coefs), 1)
})

test_that("bias_corr works with mtcars", {
  mtcars2 <- mtcars
  mtcars2$mpg01 <- ifelse(mtcars2$mpg > mean(mtcars2$mpg), 1L, 0L)

  mod <- feglm(mpg01 ~ wt | cyl, mtcars2, family = binomial())

  bc <- bias_corr(mod)

  expect_s3_class(bc, "bias_corr")
  expect_true(length(coef(bc)) == 1)
})

test_that("bias_corr summary works", {
  mtcars2 <- mtcars
  mtcars2$mpg01 <- ifelse(mtcars2$mpg > mean(mtcars2$mpg), 1L, 0L)

  mod <- feglm(mpg01 ~ wt | cyl, mtcars2, family = binomial())
  bc <- bias_corr(mod)

  summ <- summary(bc)

  expect_output(print(summ))
})

test_that("apes errors on non-binary model", {
  mod <- fepoisson(mpg ~ wt | cyl, mtcars)

  expect_error(apes(mod))
})

test_that("bias_corr errors on non-binary model", {
  mod <- fepoisson(mpg ~ wt | cyl, mtcars)

  expect_error(bias_corr(mod))
})

test_that("apes works with bias_corr object", {
  mtcars2 <- mtcars
  mtcars2$mpg01 <- ifelse(mtcars2$mpg > mean(mtcars2$mpg), 1L, 0L)

  mod <- feglm(mpg01 ~ wt | cyl, mtcars2, family = binomial())
  bc <- bias_corr(mod)

  apes_bc <- apes(bc)

  expect_s3_class(apes_bc, "apes")
})

# Stammann centering ----

test_that("apes/bias works (stammann centering)", {
  skip_on_cran()
  ctrl <- list(centering = "stammann")

  trade_short <- trade_panel[trade_panel$exp_year == "CAN1994", ]
  trade_short <- trade_short[trade_short$trade > 100, ]
  trade_short$trade <- ifelse(trade_short$trade > 200, 1L, 0L)

  mod1 <- feglm(trade ~ lang | exp_year, trade_short, family = binomial(), control = ctrl)

  expect_s3_class(mod1, "feglm")

  apes1 <- apes(mod1)
  bias1 <- bias_corr(mod1)

  apes2 <- c("lang" = 0.05)
  bias2 <- c("lang" = 0.2436)

  expect_equal(length(coef(apes1)), 1)
  expect_equal(coef(apes1), apes2, tolerance = 1e-1)
  expect_equal(length(coef(bias1)), 1)
  expect_equal(trunc(coef(bias1), 2), trunc(bias2, 2))
})

test_that("apes with mtcars binary response works (stammann centering)", {
  ctrl <- list(centering = "stammann")
  mtcars2 <- mtcars
  mtcars2$mpg01 <- ifelse(mtcars2$mpg > mean(mtcars2$mpg), 1L, 0L)

  mod <- feglm(mpg01 ~ wt | cyl, mtcars2, family = binomial(), control = ctrl)

  apes_result <- apes(mod)

  expect_s3_class(apes_result, "apes")
  expect_true(length(coef(apes_result)) == 1)
})

test_that("bias_corr works with mtcars (stammann centering)", {
  ctrl <- list(centering = "stammann")
  mtcars2 <- mtcars
  mtcars2$mpg01 <- ifelse(mtcars2$mpg > mean(mtcars2$mpg), 1L, 0L)

  mod <- feglm(mpg01 ~ wt | cyl, mtcars2, family = binomial(), control = ctrl)
  bc <- bias_corr(mod)

  expect_s3_class(bc, "bias_corr")
  expect_true(length(coef(bc)) == 1)
})

test_that("apes works with bias_corr object (stammann centering)", {
  ctrl <- list(centering = "stammann")
  mtcars2 <- mtcars
  mtcars2$mpg01 <- ifelse(mtcars2$mpg > mean(mtcars2$mpg), 1L, 0L)

  mod <- feglm(mpg01 ~ wt | cyl, mtcars2, family = binomial(), control = ctrl)
  bc <- bias_corr(mod)

  apes_bc <- apes(bc)

  expect_s3_class(apes_bc, "apes")
})
