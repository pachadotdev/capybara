test_that("fepoisson is similar to fixest", {
  mod <- fepoisson(
    trade ~ log_dist + lang + cntg + clny | exp_year + imp_year | pair,
    trade_panel
  )

  # mod_fixest <- fixest::fepois(
  #   trade ~ log_dist + lang + cntg + clny | exp_year + imp_year,
  #   trade_panel,
  #   cluster = ~pair
  # )

  coef_mod_fixest <- c(-0.8409273, 0.2474765, 0.4374432, -0.2224899)

  expect_equal(unname(round(coef(mod) - coef_mod_fixest, 5)), rep(0, 4))

  summary_mod <- summary(mod, type = "clustered")

  # the vector comes from:
  # summary_mod_fixest <- summary(mod_fixest);
  # summary_mod_fixest$coeftable[,2]
  summary_mod_fixest <- c(0.02656441, 0.06322979, 0.06825364, 0.09380935)

  expect_equal(
    unname(round(summary_mod$cm[, 2] - summary_mod_fixest, 2)),
    rep(0, 4)
  )

  expect_output(print(mod))

  expect_visible(summary(mod, type = "cluster"))

  fes <- fixed_effects(mod)
  n <- unname(mod[["nobs"]]["nobs"])
  expect_equal(length(fes), 2)
  expect_equal(length(fitted(mod)), n)
  expect_equal(length(predict(mod)), n)
  expect_equal(length(coef(mod)), 4)
  expect_equal(length(fes), 2)
  expect_equal(round(fes[["exp_year"]][1:3], 3), c(10.195, 11.081, 11.260))
  expect_equal(round(fes[["imp_year"]][1:3], 3), c(0.226, -0.254, 1.115))

  smod <- summary(mod)

  expect_equal(length(coef(smod)[, 1]), 4)
  expect_output(summary_formula_(smod))
  expect_output(summary_family_(smod))
  expect_output(summary_estimates_(smod, 3))
  expect_output(summary_r2_(smod, 3))
  expect_output(summary_nobs_(smod))
  expect_output(summary_fisher_(smod))

  trade_panel_2 <- trade_panel[trade_panel$year %in% c(2002, 2006), ]

  if (identical(Sys.info()[["user"]], "pacha")) {
    t_fepoisson <- rep(0, 10)

    t1 <- Sys.time()
    fit <- fepoisson(
      trade ~ log_dist + lang + cntg + clny | exp_year + imp_year,
      trade_panel_2
    )
    t2 <- Sys.time()
    t_fepoisson <- t2 - t1

    t_glm <- rep(0, 10)

    t1 <- Sys.time()
    fit <- suppressWarnings(glm(
      trade ~ log_dist + lang + cntg + clny + as.factor(exp_year) +
        as.factor(imp_year),
      trade_panel_2,
      family = poisson(link = "log")
    ))
    t2 <- Sys.time()

    t_glm <- t2 - t1

    expect_lte(t_fepoisson, t_glm)
  }
})
