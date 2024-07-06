test_that("fenegbin is similar to fixest", {
  mod <- fenegbin(
    trade ~ log_dist + lang + cntg + clny | exp_year + imp_year | pair,
    trade_panel
  )

  # mod_fixest <- fixest::fenegbin(
  #   trade ~ log_dist + lang + cntg + clny | exp_year + imp_year,
  #   trade_panel,
  #   cluster = ~pair
  # )

  summary_mod <- summary(mod, type = "clustered")
  
  # summary_mod_fixest <- summary(mod_fixest)
  # summary_mod_fixest$coeftable[,2][1:4]
  summary_mod_fixest <- c(0.02618568, 0.05870689, 0.12188073, 0.10366409)

  expect_equal(unname(round(summary_mod$cm[,2] - summary_mod_fixest, 1)), rep(0, 4))
})
