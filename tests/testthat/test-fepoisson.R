test_that("fepoisson is similar to fixest", {
  mod <- fepoisson(
    trade ~ log_dist + lang + cntg + clny | exp_year + imp_year | pair,
    trade_panel
  )

  mod_fixest <- fixest::fepois(
    trade ~ log_dist + lang + cntg + clny | exp_year + imp_year,
    trade_panel,
    cluster = ~pair
  )

  summary_mod <- summary(mod, type = "clustered")
  summary_mod_fixest <- summary(mod_fixest)

  expect_equal(unname(round(summary_mod$cm[,2] - summary_mod_fixest$coeftable[,2], 2)), rep(0, 4))

  expect_output(print(mod))

  expect_message(summary(mod))
  expect_visible(summary(mod, type = "cluster"))

  fes <- fixed_effects(mod)

  expect_equal(length(fes), 2)

  expect_gt(length(fitted(mod)), 0)
  expect_gt(length(predict(mod)), 0)
  expect_gt(length(coef(mod)), 0)
  expect_gt(length(coef(summary(mod))), 0)
})

