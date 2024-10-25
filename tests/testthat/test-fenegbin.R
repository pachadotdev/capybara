#' srr_stats (tests)
#'
#' @srrstatsVerbose TRUE
#'
#' @srrstats {G5.12} *Any conditions necessary to run extended tests such as platform requirements, memory, expected runtime, and artefacts produced that may need manual inspection, should be described in developer documentation such as a `CONTRIBUTING.md` or `tests/README.md` file.*
#' @noRd
NULL

test_that("fenegbin is similar to fixest", {
  # use one year or otherwise devtools::check() gives a warning about the time
  # it takes
  trade_panel_2006 <- trade_panel[trade_panel$year == 2006, ]

  mod <- fenegbin(
    trade ~ log_dist + lang + cntg + clny | exp_year + imp_year | pair,
    trade_panel_2006
  )

  # the vector comes from:
  # mod_fixest <- fixest::fenegbin(
  #   trade ~ log_dist + lang + cntg + clny | exp_year + imp_year,
  #   trade_panel_2006,
  #   cluster = ~pair
  # )

  summary_mod <- summary(mod, type = "clustered")

  # the vector comes from:
  # summary_mod_fixest <- summary(mod_fixest)
  # summary_mod_fixest$coeftable[,2][1:4]
  summary_mod_fixest <- c(0.03234993, 0.07188846, 0.14751949, 0.12471723)

  expect_equal(
    unname(round(summary_mod$cm[, 2] - summary_mod_fixest, 1)),
    rep(0, 4)
  )
})

# test_that("fenegbin time is the same adding noise to the data", {
#   trade_panel2 <- trade_panel
#   set.seed(200100)
#   trade_panel2$trade2 <- trade_panel$trade + rbinom(nrow(trade_panel2), 1, 0.5) *
#     .Machine$double.eps
#   m1 <- fenegbin(
#     trade ~ log_dist + lang + cntg + clny | exp_year + imp_year | pair,
#     trade_panel2
#   )
#   m2 <- fenegbin(
#     trade2 ~ log_dist + lang + cntg + clny | exp_year + imp_year | pair,
#     trade_panel2
#   )
#   expect_equal(coef(m1), coef(m2))
#   expect_equal(fixed_effects(m1), fixed_effects(m2))

#   t1 <- rep(NA, 10)
#   t2 <- rep(NA, 10)
#   for (i in 1:10) {
#     a <- Sys.time()
#     m1 <- fenegbin(
#       trade ~ log_dist + lang + cntg + clny | exp_year + imp_year | pair,
#       trade_panel2
#     )
#     b <- Sys.time()
#     t1[i] <- b - a

#     a <- Sys.time()
#     m2 <- fenegbin(
#       trade2 ~ log_dist + lang + cntg + clny | exp_year + imp_year | pair,
#       trade_panel2
#     )
#     b <- Sys.time()
#     t2[i] <- b - a
#   }
#   expect_lte(abs(median(t1) - median(t2)), 0.05)
# })
