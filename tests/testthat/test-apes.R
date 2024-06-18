test_that("apes works", {
  trade_short <- trade_panel[trade_panel$year %in% 2002L:2006L, ]
  trade_short$trade <- ifelse(trade_short$trade > 100, 1L, 0L)
  
  mod <- feglm(trade ~ lang | year, trade_short, family = binomial())

  expect_gt(length(coef(apes(mod))), 0)
  expect_gt(length(coef(summary(apes(mod)))), 0)
  expect_gt(length(coef(bias_corr(mod))), 0)
  expect_gt(length(coef(summary(bias_corr(mod)))), 0)
})
