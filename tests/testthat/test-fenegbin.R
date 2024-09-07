test_that("fenegbin is similar to fixest", {
  # unique(trade_panel$year)

  # use one year or otherwise devtools::check() gives a warning about the time
  # it takes
  trade_panel_2006 <- trade_panel[trade_panel$year == 2006, ]

  mod <- fenegbin(
    trade ~ log_dist + lang + cntg + clny | exp_year + imp_year | pair,
    trade_panel_2006
  )

  # mod_fixest <- fixest::fenegbin(
  #   trade ~ log_dist + lang + cntg + clny | exp_year + imp_year,
  #   trade_panel_2006,
  #   cluster = ~pair
  # )

  summary_mod <- summary(mod, type = "clustered")

  # summary_mod_fixest <- summary(mod_fixest)
  # summary_mod_fixest$coeftable[,2][1:4]
  summary_mod_fixest <- c(0.03234993, 0.07188846, 0.14751949, 0.12471723)

  expect_equal(unname(round(summary_mod$cm[, 2] - summary_mod_fixest, 1)), rep(0, 4))
})
