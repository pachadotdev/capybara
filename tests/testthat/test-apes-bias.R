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
  trade_short <- trade_panel[trade_panel$year %in% 2002L:2006L, ]
  trade_short$trade <- ifelse(trade_short$trade > 100, 1L, 0L)

  mod1 <- feglm(trade ~ lang | year, trade_short, family = binomial())
  apes1 <- apes(mod1)
  bias1 <- bias_corr(mod1)

  # the values come from:
  # mod2 <- alpaca::feglm(trade ~ lang | year, trade_short, family = binomial())
  # apes2 <- alpaca::getAPEs(mod2)
  # bias2 <- alpaca::biasCorr(mod2)
  apes2 <- c("lang" = 0.05594)
  bias2 <- c("lang" = 0.23390)

  expect_output(print(mod1))

  expect_equal(length(coef(apes1)), 1)
  expect_equal(round(coef(apes1), 3), round(apes2, 3))
  expect_equal(length(coef(summary(apes(mod1)))), 4)

  expect_equal(length(coef(bias1)), 1)
  expect_equal(round(coef(bias1), 1), round(bias2, 1))
  expect_equal(length(coef(summary(bias1))), 4)
})
