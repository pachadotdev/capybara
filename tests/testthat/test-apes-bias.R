test_that("apes/bias works", {
  trade_short <- trade_panel[trade_panel$year %in% 2002L:2006L, ]
  trade_short$trade <- ifelse(trade_short$trade > 100, 1L, 0L)

  mod1 <- feglm(trade ~ lang | year, trade_short, family = binomial())
  apes1 <- apes(mod1)
  bias1 <- bias_corr(mod1)

  # mod2 <- alpaca::feglm(trade ~ lang | year, trade_short, family = binomial())
  # apes2 <- alpaca::getAPEs(mod2)
  # bias2 <- alpaca::biasCorr(mod2)
  apes2 <- c("lang" = 0.05594)
  bias2 <- c("lang" = 0.23390)

  expect_output(print(mod1))

  expect_equal(length(coef(apes1)), 1)
  expect_equal(round(coef(apes1), 5), apes2)
  expect_equal(length(coef(summary(apes(mod1)))), 4)

  expect_equal(length(coef(bias1)), 1)
  expect_equal(round(coef(bias1), 1), round(bias2, 1))
  expect_equal(length(coef(summary(bias1))), 4)
})
