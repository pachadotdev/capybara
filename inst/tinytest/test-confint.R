# srr_stats (tests)
# {G1.0} Implements unit testing for confidence intervals.
# {G2.3} Tests compatibility with R generics conventions.
# {RE3.1} Verifies the correctness of extracted model statistics.

local({
  if (Sys.getenv("CAPYBARA_FULL_TESTING") != "yes") {
    return(NULL)
  }

  # confint.feglm returns correct structure and values ----

  ross2004_subset <- ross2004[ross2004$year == 1999, ]
  ross2004_subset <- ross2004_subset[ross2004_subset$ltrade > quantile(ross2004_subset$ltrade, 0.75), ]

  mod1 <- fepoisson(ltrade ~ ldist | ctry1, ross2004_subset)

  res1 <- confint(mod1)

  # Check structure
  expect_equal(ncol(res1), 2)
  expect_equal(nrow(res1), 1)
  expect_equal(rownames(res1), "ldist")

  # Manually compute Wald CI to verify correctness
  est <- mod1$coef_table["ldist", "Estimate"]
  se <- mod1$coef_table["ldist", "Std. Error"]
  z <- qnorm(0.975)
  expected_ci <- matrix(c(est - z * se, est + z * se), nrow = 1)
  colnames(expected_ci) <- c("2.5 %", "97.5 %")
  rownames(expected_ci) <- "ldist"

  expect_equal(res1, expected_ci, tolerance = 1e-10)

  # Check that CI is symmetric around estimate
  midpoint <- (res1[1, 1] + res1[1, 2]) / 2
  expect_equal(midpoint, est, tolerance = 1e-10)

  # confint.feglm respects level parameter ----

  mod <- fepoisson(ltrade ~ ldist | ctry1, ross2004_subset)

  ci_95 <- confint(mod, level = 0.95)
  ci_99 <- confint(mod, level = 0.99)

  # 99% CI should be wider
  width_95 <- ci_95[, 2] - ci_95[, 1]
  width_99 <- ci_99[, 2] - ci_99[, 1]

  expect_true(all(width_99 > width_95))

  # confint.felm returns correct structure ----

  mod <- felm(ltrade ~ ldist | ctry1, ross2004_subset)

  result <- confint(mod)

  expect_equal(ncol(result), 2)
  expect_equal(nrow(result), length(coef(mod)))

  # confint column names reflect confidence level ----

  mod <- felm(ltrade ~ ldist | ctry1, ross2004_subset)

  ci_95 <- confint(mod, level = 0.95)
  ci_90 <- confint(mod, level = 0.90)

  expect_true(grepl("2.5", colnames(ci_95)[1]))
  expect_true(grepl("97.5", colnames(ci_95)[2]))
  expect_true(grepl("5", colnames(ci_90)[1]))
  expect_true(grepl("95", colnames(ci_90)[2]))

  # confint works with multiple parm selection ----

  mod <- felm(ltrade ~ ldist + border | ctry1, ross2004_subset)

  ci_subset <- confint(mod, parm = c("ldist", "border"))

  expect_equal(nrow(ci_subset), 2)
  expect_equal(rownames(ci_subset), c("ldist", "border"))

  # confint works with numeric parm indices ----

  mod <- felm(ltrade ~ ldist + border | ctry1, ross2004_subset)

  ci_first <- confint(mod, parm = 1)

  expect_equal(nrow(ci_first), 1)
  expect_equal(rownames(ci_first), "ldist")

  # confint for feglm works with parm ----

  mod <- fepoisson(ltrade ~ ldist | ctry1, ross2004_subset)

  ci_full <- confint(mod)
  ci_parm <- confint(mod, parm = "ldist")

  expect_equal(ci_full, ci_parm)

  # confint handles different confidence levels correctly ----

  mod <- felm(ltrade ~ ldist | ctry1, ross2004_subset)

  ci_50 <- confint(mod, level = 0.50)
  ci_90 <- confint(mod, level = 0.90)
  ci_99 <- confint(mod, level = 0.99)

  width_50 <- ci_50[1, 2] - ci_50[1, 1]
  width_90 <- ci_90[1, 2] - ci_90[1, 1]
  width_99 <- ci_99[1, 2] - ci_99[1, 1]

  expect_true(width_50 < width_90)
  expect_true(width_90 < width_99)
})
