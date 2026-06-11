# srr_stats (tests)
# {G1.0} Implements unit testing for broom generics (tidy, glance, augment).
# {G2.3} Tests compatibility with broom package conventions.
# {RE3.1} Verifies the correctness of extracted model statistics.

local({
  if (Sys.getenv("CAPYBARA_FULL_TESTING") != "yes") {
    return(NULL)
  }

  # glance.feglm returns correct structure ----

  ross2004_subset <- ross2004[ross2004$year == 1999, ]
  ross2004_subset <- ross2004_subset[ross2004_subset$ltrade > quantile(ross2004_subset$ltrade, 0.75), ]

  mod <- fepoisson(ltrade ~ ldist | ctry1, ross2004_subset)

  result <- glance(mod)

  expect_true(is.data.frame(result))
  expect_true("deviance" %in% names(result))
  expect_true("null_deviance" %in% names(result))
  expect_true("nobs" %in% names(result))

  # glance.felm returns correct structure ----

  mod <- felm(ltrade ~ ldist | ctry1, ross2004_subset)

  result <- glance(mod)

  expect_true(is.data.frame(result))
  expect_true("r_squared" %in% names(result))
  expect_true("adj_r_squared" %in% names(result))
  expect_true("nobs" %in% names(result))

  # glance.felm works with multiple fixed effects ----

  mod <- felm(ltrade ~ ldist | ctry1 + ctry2, ross2004_subset)

  result <- glance(mod)

  expect_true(is.data.frame(result))
  expect_true(result$r_squared > 0 && result$r_squared < 1)

  # tidy.feglm returns correct structure ----

  mod <- fepoisson(ltrade ~ ldist | ctry1, ross2004_subset)

  result <- tidy(mod)

  expect_true(is.data.frame(result))
  expect_equal(
    names(result),
    c("estimate", "std.error", "statistic", "p.value")
  )

  # tidy.feglm works with conf_int ----

  mod <- fepoisson(ltrade ~ ldist | ctry1, ross2004_subset)

  result <- tidy(mod, conf_int = TRUE)

  expect_true(is.data.frame(result))
  expect_true("conf.low" %in% names(result))
  expect_true("conf.high" %in% names(result))
  expect_true(all(result$conf.low < result$estimate))
  expect_true(all(result$conf.high > result$estimate))

  # tidy.feglm respects conf_level ----

  mod <- fepoisson(ltrade ~ ldist | ctry1, ross2004_subset)

  result_95 <- tidy(mod, conf_int = TRUE, conf_level = 0.95)
  result_99 <- tidy(mod, conf_int = TRUE, conf_level = 0.99)

  # 99% CI should be wider than 95% CI
  width_95 <- result_95$conf.high - result_95$conf.low
  width_99 <- result_99$conf.high - result_99$conf.low

  expect_true(all(width_99 > width_95))

  # tidy.felm returns correct structure ----

  mod <- felm(ltrade ~ ldist | ctry1, ross2004_subset)

  result <- tidy(mod)

  expect_true(is.data.frame(result))
  expect_equal(
    names(result),
    c("estimate", "std.error", "statistic", "p.value")
  )

  # tidy.felm works with conf_int ----

  mod <- felm(ltrade ~ ldist | ctry1, ross2004_subset)

  result <- tidy(mod, conf_int = TRUE)

  expect_true(is.data.frame(result))
  expect_true("conf.low" %in% names(result))
  expect_true("conf.high" %in% names(result))

  # tidy works with multiple predictors ----

  mod <- felm(ltrade ~ ldist + border + comlang | ctry1, ross2004_subset)

  result <- tidy(mod)

  expect_equal(nrow(result), 3)

  # augment.feglm returns correct structure ----

  mod <- fepoisson(ltrade ~ ldist | ctry1, ross2004_subset, control = fit_control(keep_data = TRUE))

  result <- augment(mod)

  expect_true(is.data.frame(result))
  expect_true(".fitted" %in% names(result))
  expect_true(".residuals" %in% names(result))
  expect_equal(nrow(result), nrow(ross2004_subset))

  # augment.feglm preserves original columns ----

  mod <- fepoisson(ltrade ~ ldist | ctry1, ross2004_subset, control = fit_control(keep_data = TRUE))

  result <- augment(mod)

  expect_true("ltrade" %in% names(result))
  expect_true("ldist" %in% names(result))
  expect_true("ctry1" %in% names(result))

  # augment.felm returns correct structure ----

  mod <- felm(ltrade ~ ldist | ctry1, ross2004_subset, control = fit_control(keep_data = TRUE))

  result <- augment(mod)

  expect_true(is.data.frame(result))
  expect_true(".fitted" %in% names(result))
  expect_true(".residuals" %in% names(result))

  # augment.felm fitted values are reasonable ----

  mod <- felm(ltrade ~ ldist | ctry1, ross2004_subset, control = fit_control(keep_data = TRUE))

  result <- augment(mod)

  # Fitted values should be in a reasonable range (log-transformed ltrade)
  expect_true(all(is.finite(result$.fitted)))
  expect_true(length(result$.fitted) > 0)

  # fitted.feglm returns correct values ----

  mod <- fepoisson(ltrade ~ ldist | ctry1, ross2004_subset)

  result <- fitted(mod)

  expect_equal(length(result), nrow(ross2004_subset))
  expect_true(all(result > 0))

  # fitted.felm returns correct values ----

  mod <- felm(ltrade ~ ldist | ctry1, ross2004_subset)

  result <- fitted(mod)

  expect_equal(length(result), nrow(ross2004_subset))
})
