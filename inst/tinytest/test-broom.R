#' srr_stats (tests)
#' @srrstats {G1.0} Implements unit testing for broom generics (tidy, glance, augment).
#' @srrstats {G2.3} Tests compatibility with broom package conventions.
#' @srrstats {RE3.1} Verifies the correctness of extracted model statistics.
#' @noRd
NULL

# ---- glance tests ----

# glance.feglm returns correct structure
local({
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  
  mod <- fepoisson(trade ~ log_dist | exp_year, yotov2017_subset)

  result <- glance(mod)

  expect_true(is.data.frame(result))
  expect_true("deviance" %in% names(result))
  expect_true("null_deviance" %in% names(result))
  expect_true("nobs" %in% names(result))
})

# glance.felm returns correct structure
local({
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  
  mod <- felm(trade ~ log_dist | exp_year, yotov2017_subset)

  result <- glance(mod)

  expect_true(is.data.frame(result))
  expect_true("r_squared" %in% names(result))
  expect_true("adj_r_squared" %in% names(result))
  expect_true("nobs" %in% names(result))
})

# glance.felm works with multiple fixed effects
local({
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]

  mod <- felm(trade ~ log_dist | exp_year + imp_year, yotov2017_subset)

  result <- glance(mod)

  expect_true(is.data.frame(result))
  expect_true(result$r_squared > 0 && result$r_squared < 1)
})

# ---- tidy tests ----

# tidy.feglm returns correct structure
local({
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  
  mod <- fepoisson(trade ~ log_dist | exp_year, yotov2017_subset)

  result <- tidy(mod)

  expect_true(is.data.frame(result))
  expect_equal(
    names(result),
    c("estimate", "std.error", "statistic", "p.value")
  )
})

# tidy.feglm works with conf_int
local({
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  
  mod <- fepoisson(trade ~ log_dist | exp_year, yotov2017_subset)

  result <- tidy(mod, conf_int = TRUE)

  expect_true(is.data.frame(result))
  expect_true("conf.low" %in% names(result))
  expect_true("conf.high" %in% names(result))
  expect_true(all(result$conf.low < result$estimate))
  expect_true(all(result$conf.high > result$estimate))
})

# tidy.feglm respects conf_level
local({
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]

  mod <- fepoisson(trade ~ log_dist | exp_year, yotov2017_subset)

  result_95 <- tidy(mod, conf_int = TRUE, conf_level = 0.95)
  result_99 <- tidy(mod, conf_int = TRUE, conf_level = 0.99)

  # 99% CI should be wider than 95% CI
  width_95 <- result_95$conf.high - result_95$conf.low
  width_99 <- result_99$conf.high - result_99$conf.low

  expect_true(all(width_99 > width_95))
})

# tidy.felm returns correct structure
local({
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  
  mod <- felm(trade ~ log_dist | exp_year, yotov2017_subset)

  result <- tidy(mod)

  expect_true(is.data.frame(result))
  expect_equal(
    names(result),
    c("estimate", "std.error", "statistic", "p.value")
  )
})

# tidy.felm works with conf_int
local({
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  
  mod <- felm(trade ~ log_dist | exp_year, yotov2017_subset)

  result <- tidy(mod, conf_int = TRUE)

  expect_true(is.data.frame(result))
  expect_true("conf.low" %in% names(result))
  expect_true("conf.high" %in% names(result))
})

# tidy works with multiple predictors
local({
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]

  mod <- felm(trade ~ log_dist + cntg + lang | exp_year, yotov2017_subset)

  result <- tidy(mod)

  expect_equal(nrow(result), 3)
})

# ---- augment tests ----

# augment.feglm returns correct structure
local({
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]

  mod <- fepoisson(trade ~ log_dist | exp_year, yotov2017_subset, control = fit_control(keep_data = TRUE))

  result <- augment(mod)

  expect_true(is.data.frame(result))
  expect_true(".fitted" %in% names(result))
  expect_true(".residuals" %in% names(result))
  expect_equal(nrow(result), nrow(yotov2017_subset))
})

# augment.feglm preserves original columns
local({
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]

  mod <- fepoisson(trade ~ log_dist | exp_year, yotov2017_subset, control = fit_control(keep_data = TRUE))

  result <- augment(mod)

  expect_true("trade" %in% names(result))
  expect_true("log_dist" %in% names(result))
  expect_true("exp_year" %in% names(result))
})

# augment.felm returns correct structure
local({
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]

  mod <- felm(trade ~ log_dist | exp_year, yotov2017_subset, control = fit_control(keep_data = TRUE))

  result <- augment(mod)

  expect_true(is.data.frame(result))
  expect_true(".fitted" %in% names(result))
  expect_true(".residuals" %in% names(result))
})

# augment.felm fitted values are reasonable
local({
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]

  mod <- felm(trade ~ log_dist | exp_year, yotov2017_subset, control = fit_control(keep_data = TRUE))

  result <- augment(mod)

  # Fitted values should be in a reasonable range (log-transformed trade)
  expect_true(all(is.finite(result$.fitted)))
  expect_true(length(result$.fitted) > 0)
})

# ---- fitted tests ----

# fitted.feglm returns correct values
local({
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]

  mod <- fepoisson(trade ~ log_dist | exp_year, yotov2017_subset)

  result <- fitted(mod)

  expect_equal(length(result), nrow(yotov2017_subset))
  expect_true(all(result > 0))
})

# fitted.felm returns correct values
local({
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]

  mod <- felm(trade ~ log_dist | exp_year, yotov2017_subset)

  result <- fitted(mod)

  expect_equal(length(result), nrow(yotov2017_subset))
})
