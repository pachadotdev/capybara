#' srr_stats (tests)
#' @srrstats {G2.11} Tests for `data.frame`-like tabular objects with unit-type columns.
#' @noRd
NULL

test_that("feglm handles unit-type columns", {
  set.seed(123)

  # F = m * a
  # log(F) = log(m) * log(a)
  # add noise to the data

  d <- data.frame(
    log_m = log(abs(units::as_units(rnorm(100), "m"))),
    log_a = log(abs(units::as_units(rnorm(100), "m/s^2"))),
    lab = factor(rep(1:10, 10))
  )

  d$log_f <- d$log_m * d$log_a

  d$log_f <- d$log_f + abs(rnorm(100, 0, 0.1)) * d$log_f

  mod <- feglm(log_f ~ log_m + log_a | lab, d)

  expect_error(
    glm(
      log_f ~ log_m + log_a,
      d,
      family = gaussian()
    ),
    "should be \"units\" objects"
  )

  d2 <- d

  d2$log_f <- as.numeric(d2$log_f)
  d2$log_m <- as.numeric(d2$log_m)
  d2$log_a <- as.numeric(d2$log_a)

  mod_base <- glm(
    log_f ~ log_m + log_a + as.factor(lab),
    d2,
    family = gaussian()
  )

  expect_equal(
    round(coef(mod)["log_m"], 2),
    round(coef(mod_base)["log_m"], 2)
  )

  expect_equal(
    round(coef(mod)["log_a"], 2),
    round(coef(mod_base)["log_a"], 2)
  )
})
