#' srr_stats (tests)
#' @srrstats {G1.0} Implements unit testing for print functionality.
#' @srrstats {G3.2} Verifies correct print output format.
#' @srrstats {RE4.17} Validates default print methods for models and summaries.
#' @noRd
NULL

# ---- print.feglm tests ----

test_that("print.feglm produces output", {
  mod <- fepoisson(mpg ~ wt | cyl, mtcars)

  expect_output(print(mod))
})

test_that("print.feglm shows coefficients", {
  mod <- fepoisson(mpg ~ wt | cyl, mtcars)

  output <- capture.output(print(mod))

  expect_true(any(grepl("wt", output)))
})

# ---- print.felm tests ----

test_that("print.felm produces output", {
  mod <- felm(mpg ~ wt | cyl, mtcars)

  expect_output(print(mod))
})

test_that("print.felm shows coefficients", {
  mod <- felm(mpg ~ wt | cyl, mtcars)

  output <- capture.output(print(mod))

  expect_true(any(grepl("wt", output)))
})

# ---- print.summary.feglm tests ----

test_that("print.summary.feglm produces output", {
  mod <- fepoisson(mpg ~ wt | cyl, mtcars)

  expect_output(print(summary(mod)))
})

test_that("summary.feglm shows formula", {
  mod <- fepoisson(mpg ~ wt | cyl, mtcars)

  output <- capture.output(print(summary(mod)))

  expect_true(any(grepl("Formula", output)))
})

test_that("summary.feglm shows family", {
  mod <- fepoisson(mpg ~ wt | cyl, mtcars)

  output <- capture.output(print(summary(mod)))

  expect_true(any(
    grepl("Family", output) | grepl("poisson", output, ignore.case = TRUE)
  ))
})

test_that("summary.feglm shows estimates", {
  mod <- fepoisson(mpg ~ wt | cyl, mtcars)

  output <- capture.output(print(summary(mod)))

  expect_true(any(grepl("Estimate", output)))
})

test_that("summary.feglm shows significance codes", {
  mod <- fepoisson(mpg ~ wt | cyl, mtcars)

  output <- capture.output(print(summary(mod)))

  expect_true(any(grepl("Significance", output)))
})

# ---- print.summary.felm tests ----

test_that("print.summary.felm produces output", {
  mod <- felm(mpg ~ wt | cyl, mtcars)

  expect_output(print(summary(mod)))
})

test_that("summary.felm shows R-squared", {
  mod <- felm(mpg ~ wt | cyl, mtcars)

  output <- capture.output(print(summary(mod)))

  expect_true(any(grepl("R-squared|R²", output)))
})

# ---- print with multiple predictors ----

test_that("print shows multiple predictors", {
  mod <- felm(mpg ~ wt + hp + qsec | cyl, mtcars)

  output <- capture.output(print(summary(mod)))

  expect_true(any(grepl("wt", output)))
  expect_true(any(grepl("hp", output)))
  expect_true(any(grepl("qsec", output)))
})

# ---- print for binomial ----

test_that("print works for binomial model", {
  mod <- feglm(am ~ wt | cyl, mtcars, family = binomial())

  expect_output(print(mod))
  expect_output(print(summary(mod)))
})

# ---- print.apes tests ----

test_that("print.apes produces output", {
  mtcars2 <- mtcars
  mtcars2$mpg01 <- ifelse(mtcars2$mpg > mean(mtcars2$mpg), 1L, 0L)

  mod <- feglm(mpg01 ~ wt | cyl, mtcars2, family = binomial())
  apes_result <- apes(mod)

  expect_output(print(apes_result))
})

test_that("print.summary.apes produces output", {
  mtcars2 <- mtcars
  mtcars2$mpg01 <- ifelse(mtcars2$mpg > mean(mtcars2$mpg), 1L, 0L)

  mod <- feglm(mpg01 ~ wt | cyl, mtcars2, family = binomial())
  apes_result <- apes(mod)

  expect_output(print(summary(apes_result)))
})

# ---- print.bias_corr tests ----

test_that("print.bias_corr produces output", {
  mtcars2 <- mtcars
  mtcars2$mpg01 <- ifelse(mtcars2$mpg > mean(mtcars2$mpg), 1L, 0L)

  mod <- feglm(mpg01 ~ wt | cyl, mtcars2, family = binomial())
  bc <- bias_corr(mod)

  expect_output(print(bc))
})

test_that("print.summary.bias_corr produces output", {
  mtcars2 <- mtcars
  mtcars2$mpg01 <- ifelse(mtcars2$mpg > mean(mtcars2$mpg), 1L, 0L)

  mod <- feglm(mpg01 ~ wt | cyl, mtcars2, family = binomial())
  bc <- bias_corr(mod)

  expect_output(print(summary(bc)))
})

# ---- print for models without FE ----

test_that("print works for model without fixed effects", {
  mod <- fepoisson(mpg ~ wt, mtcars)

  expect_output(print(mod))
  expect_output(print(summary(mod)))
})

test_that("print.felm works for model without fixed effects", {
  mod <- felm(mpg ~ wt, mtcars)

  expect_output(print(mod))
  expect_output(print(summary(mod)))
})
