#' srr_stats (tests)
#' @srrstats {G1.0} Implements unit testing for print functionality.
#' @srrstats {G3.2} Verifies correct print output format.
#' @srrstats {RE4.17} Validates default print methods for models and summaries.
#' @noRd
NULL

# ---- print.feglm tests ----

# print.feglm shows coefficients"
local({
  mod <- fepoisson(mpg ~ wt | cyl, mtcars)

  output <- capture.output(print(mod))

  expect_true(any(grepl("wt", output)))
})

# ---- print.felm tests ----

# print.felm shows coefficients"
local({
  mod <- felm(mpg ~ wt | cyl, mtcars)

  output <- capture.output(print(mod))

  expect_true(any(grepl("wt", output)))
})

# ---- print.summary.feglm tests ----

# summary.feglm shows formula"
local({
  mod <- fepoisson(mpg ~ wt | cyl, mtcars)

  output <- capture.output(print(summary(mod)))

  expect_true(any(grepl("Formula", output)))
})

# summary.feglm shows family"
local({
  mod <- fepoisson(mpg ~ wt | cyl, mtcars)

  output <- capture.output(print(summary(mod)))

  expect_true(any(
    grepl("Family", output) | grepl("poisson", output, ignore.case = TRUE)
  ))
})

# summary.feglm shows estimates"
local({
  mod <- fepoisson(mpg ~ wt | cyl, mtcars)

  output <- capture.output(print(summary(mod)))

  expect_true(any(grepl("Estimate", output)))
})

# summary.feglm shows significance codes"
local({
  mod <- fepoisson(mpg ~ wt | cyl, mtcars)

  output <- capture.output(print(summary(mod)))

  expect_true(any(grepl("Significance", output)))
})

# ---- print.summary.felm tests ----

# summary.felm shows R-squared"
local({
  mod <- felm(mpg ~ wt | cyl, mtcars)

  output <- capture.output(print(summary(mod)))

  expect_true(any(grepl("R-squared|R²", output)))
})

# ---- print with multiple predictors ----

# print shows multiple predictors"
local({
  mod <- felm(mpg ~ wt + hp + qsec | cyl, mtcars)

  output <- capture.output(print(summary(mod)))

  expect_true(any(grepl("wt", output)))
  expect_true(any(grepl("hp", output)))
  expect_true(any(grepl("qsec", output)))
})
