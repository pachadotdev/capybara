#' srr_stats (tests)
#' @srrstats {G1.0} Implements unit testing for broom generics (tidy, glance, augment).
#' @srrstats {G2.3} Tests compatibility with broom package conventions.
#' @srrstats {RE3.1} Verifies the correctness of extracted model statistics.
#' @noRd
NULL

# ---- glance tests ----

# glance.feglm returns correct structure
local({
  mod <- fepoisson(mpg ~ wt | cyl, mtcars)

  result <- glance(mod)

  expect_true(is.data.frame(result))
  expect_true("deviance" %in% names(result))
  expect_true("null_deviance" %in% names(result))
  expect_true("nobs" %in% names(result))
})

# glance.feglm works with binomial
local({
  mod <- feglm(am ~ wt | cyl, mtcars, family = binomial())

  result <- glance(mod)

  expect_true(is.data.frame(result))
  expect_true(is.numeric(result$deviance))
})

# glance.felm returns correct structure
local({
  mod <- felm(mpg ~ wt | cyl, mtcars)

  result <- glance(mod)

  expect_true(is.data.frame(result))
  expect_true("r_squared" %in% names(result))
  expect_true("adj_r_squared" %in% names(result))
  expect_true("nobs" %in% names(result))
})

# glance.felm works with multiple fixed effects
local({
  mod <- felm(mpg ~ wt | cyl + am, mtcars)

  result <- glance(mod)

  expect_true(is.data.frame(result))
  expect_true(result$r_squared > 0 && result$r_squared < 1)
})

# ---- tidy tests ----

# tidy.feglm returns correct structure
local({
  mod <- fepoisson(mpg ~ wt | cyl, mtcars)

  result <- tidy(mod)

  expect_true(is.data.frame(result))
  expect_equal(
    names(result),
    c("estimate", "std.error", "statistic", "p.value")
  )
})

# tidy.feglm works with conf_int
local({
  mod <- fepoisson(mpg ~ wt | cyl, mtcars)

  result <- tidy(mod, conf_int = TRUE)

  expect_true(is.data.frame(result))
  expect_true("conf.low" %in% names(result))
  expect_true("conf.high" %in% names(result))
  expect_true(all(result$conf.low < result$estimate))
  expect_true(all(result$conf.high > result$estimate))
})

# tidy.feglm respects conf_level
local({
  mod <- fepoisson(mpg ~ wt | cyl, mtcars)

  result_95 <- tidy(mod, conf_int = TRUE, conf_level = 0.95)
  result_99 <- tidy(mod, conf_int = TRUE, conf_level = 0.99)

  # 99% CI should be wider than 95% CI
  width_95 <- result_95$conf.high - result_95$conf.low
  width_99 <- result_99$conf.high - result_99$conf.low

  expect_true(all(width_99 > width_95))
})

# tidy.felm returns correct structure
local({
  mod <- felm(mpg ~ wt | cyl, mtcars)

  result <- tidy(mod)

  expect_true(is.data.frame(result))
  expect_equal(
    names(result),
    c("estimate", "std.error", "statistic", "p.value")
  )
})

# tidy.felm works with conf_int
local({
  mod <- felm(mpg ~ wt | cyl, mtcars)

  result <- tidy(mod, conf_int = TRUE)

  expect_true(is.data.frame(result))
  expect_true("conf.low" %in% names(result))
  expect_true("conf.high" %in% names(result))
})

# tidy works with multiple predictors
local({
  mod <- felm(mpg ~ wt + hp + qsec | cyl, mtcars)

  result <- tidy(mod)

  expect_equal(nrow(result), 3)
})

# ---- augment tests ----

# augment.feglm returns correct structure
local({
  mod <- fepoisson(mpg ~ wt | cyl, mtcars, control = fit_control(keep_data = TRUE))

  result <- augment(mod)

  expect_true(is.data.frame(result))
  expect_true(".fitted" %in% names(result))
  expect_true(".residuals" %in% names(result))
  expect_equal(nrow(result), nrow(mtcars))
})

# augment.feglm preserves original columns
local({
  mod <- fepoisson(mpg ~ wt | cyl, mtcars, control = fit_control(keep_data = TRUE))

  result <- augment(mod)

  expect_true("mpg" %in% names(result))
  expect_true("wt" %in% names(result))
  expect_true("cyl" %in% names(result))
})

# augment.felm returns correct structure
local({
  mod <- felm(mpg ~ wt | cyl, mtcars, control = fit_control(keep_data = TRUE))

  result <- augment(mod)

  expect_true(is.data.frame(result))
  expect_true(".fitted" %in% names(result))
  expect_true(".residuals" %in% names(result))
})

# augment.felm fitted values are reasonable
local({
  mod <- felm(mpg ~ wt | cyl, mtcars, control = fit_control(keep_data = TRUE))

  result <- augment(mod)

  # Fitted values should be in a reasonable range
  expect_true(all(result$.fitted > 0))
  expect_true(all(result$.fitted < 50))
})

# augment works with binomial model
local({
  mod <- feglm(am ~ wt | cyl, mtcars, family = binomial(), control = fit_control(keep_data = TRUE))

  result <- augment(mod)

  expect_true(is.data.frame(result))
  expect_true(".fitted" %in% names(result))
  # Fitted values for binomial should be probabilities
  expect_true(all(result$.fitted >= 0 & result$.fitted <= 1))
})

# ---- fitted tests ----

# fitted.feglm returns correct values
local({
  mod <- fepoisson(mpg ~ wt | cyl, mtcars)

  result <- fitted(mod)

  expect_equal(length(result), nrow(mtcars))
  expect_true(all(result > 0))
})

# fitted.felm returns correct values
local({
  mod <- felm(mpg ~ wt | cyl, mtcars)

  result <- fitted(mod)

  expect_equal(length(result), nrow(mtcars))
})
