#' srr_stats (tests)
#' @srrstats {G1.0} Implements unit testing for confidence intervals.
#' @srrstats {G2.3} Tests compatibility with R generics conventions.
#' @srrstats {RE3.1} Verifies the correctness of extracted model statistics.
#' @noRd
NULL

# ---- confint tests ----

test_that("confint.feglm returns correct structure and values", {
  mod1 <- fepoisson(mpg ~ wt | cyl, mtcars)

  res1 <- confint(mod1)

  # Check structure
  expect_equal(ncol(res1), 2)
  expect_equal(nrow(res1), 1)
  expect_equal(rownames(res1), "wt")

  # Manually compute Wald CI to verify correctness
  est <- mod1$coef_table["wt", "Estimate"]
  se <- mod1$coef_table["wt", "Std. Error"]
  z <- qnorm(0.975)
  expected_ci <- matrix(c(est - z * se, est + z * se), nrow = 1)
  colnames(expected_ci) <- c("2.5 %", "97.5 %")
  rownames(expected_ci) <- "wt"

  expect_equal(res1, expected_ci, tolerance = 1e-10)

  # Check that CI is symmetric around estimate
  midpoint <- (res1[1, 1] + res1[1, 2]) / 2
  expect_equal(midpoint, est, tolerance = 1e-10)
})

test_that("confint.feglm respects level parameter", {
  mod <- fepoisson(mpg ~ wt | cyl, mtcars)

  ci_95 <- confint(mod, level = 0.95)
  ci_99 <- confint(mod, level = 0.99)

  # 99% CI should be wider
  width_95 <- ci_95[, 2] - ci_95[, 1]
  width_99 <- ci_99[, 2] - ci_99[, 1]

  expect_true(all(width_99 > width_95))
})

test_that("confint.felm returns correct structure", {
  mod <- felm(mpg ~ wt | cyl, mtcars)

  result <- confint(mod)

  expect_equal(ncol(result), 2)
  expect_equal(nrow(result), length(coef(mod)))
})

test_that("confint column names reflect confidence level", {
  mod <- felm(mpg ~ wt | cyl, mtcars)

  ci_95 <- confint(mod, level = 0.95)
  ci_90 <- confint(mod, level = 0.90)

  expect_true(grepl("2.5", colnames(ci_95)[1]))
  expect_true(grepl("97.5", colnames(ci_95)[2]))
  expect_true(grepl("5", colnames(ci_90)[1]))
  expect_true(grepl("95", colnames(ci_90)[2]))
})

test_that("confint works with parm parameter", {
  mod <- felm(mpg ~ wt + hp + qsec | cyl, mtcars)

  # Select specific parameters
  ci_wt <- confint(mod, parm = "wt")

  expect_equal(nrow(ci_wt), 1)
  expect_equal(rownames(ci_wt), "wt")
})

test_that("confint works with multiple parm selection", {
  mod <- felm(mpg ~ wt + hp + qsec | cyl, mtcars)

  ci_subset <- confint(mod, parm = c("wt", "hp"))

  expect_equal(nrow(ci_subset), 2)
  expect_equal(rownames(ci_subset), c("wt", "hp"))
})

test_that("confint works with numeric parm indices", {
  mod <- felm(mpg ~ wt + hp + qsec | cyl, mtcars)

  ci_first <- confint(mod, parm = 1)

  expect_equal(nrow(ci_first), 1)
  expect_equal(rownames(ci_first), "wt")
})

test_that("confint for feglm works with parm", {
  mod <- fepoisson(mpg ~ wt | cyl, mtcars)

  ci_full <- confint(mod)
  ci_parm <- confint(mod, parm = "wt")

  expect_equal(ci_full, ci_parm)
})

test_that("confint handles different confidence levels correctly", {
  mod <- felm(mpg ~ wt | cyl, mtcars)

  ci_50 <- confint(mod, level = 0.50)
  ci_90 <- confint(mod, level = 0.90)
  ci_99 <- confint(mod, level = 0.99)

  width_50 <- ci_50[1, 2] - ci_50[1, 1]
  width_90 <- ci_90[1, 2] - ci_90[1, 1]
  width_99 <- ci_99[1, 2] - ci_99[1, 1]

  expect_true(width_50 < width_90)
  expect_true(width_90 < width_99)
})
