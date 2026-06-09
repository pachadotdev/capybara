#' srr_stats (tests)
#' @srrstats {G1.0} Implements unit testing for print functionality.
#' @srrstats {G3.2} Verifies correct print output format.
#' @srrstats {RE4.17} Validates default print methods for models and summaries.
#' @noRd
NULL

# ---- print.feglm tests ----

# print.feglm shows coefficients
local({
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  
  mod <- fepoisson(trade ~ log_dist | exp_year, yotov2017_subset)

  output <- capture.output(print(mod))

  expect_true(any(grepl("log_dist", output)))
})

# ---- print.felm tests ----

# print.felm shows coefficients
local({
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  
  mod <- felm(trade ~ log_dist | exp_year, yotov2017_subset)

  output <- capture.output(print(mod))

  expect_true(any(grepl("log_dist", output)))
})

# ---- print.summary.feglm tests ----

# summary.feglm shows formula
local({
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  
  mod <- fepoisson(trade ~ log_dist | exp_year, yotov2017_subset)

  output <- capture.output(print(summary(mod)))

  expect_true(any(grepl("Formula", output)))
})

# summary.feglm shows family
local({
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  
  mod <- fepoisson(trade ~ log_dist | exp_year, yotov2017_subset)

  output <- capture.output(print(summary(mod)))

  expect_true(any(
    grepl("Family", output) | grepl("poisson", output, ignore.case = TRUE)
  ))
})

# summary.feglm shows estimates
local({
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  
  mod <- fepoisson(trade ~ log_dist | exp_year, yotov2017_subset)

  output <- capture.output(print(summary(mod)))

  expect_true(any(grepl("Estimate", output)))
})

# summary.feglm shows significance codes
local({
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  
  mod <- fepoisson(trade ~ log_dist | exp_year, yotov2017_subset)

  output <- capture.output(print(summary(mod)))

  expect_true(any(grepl("Significance", output)))
})

# ---- print.summary.felm tests ----

# summary.felm shows R-squared
local({
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  
  mod <- felm(trade ~ log_dist | exp_year, yotov2017_subset)

  output <- capture.output(print(summary(mod)))

  expect_true(any(grepl("R-squared|R²", output)))
})

# ---- print with multiple predictors ----

# print shows multiple predictors
local({
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  
  mod <- felm(trade ~ log_dist + cntg | exp_year, yotov2017_subset)

  output <- capture.output(print(summary(mod)))

  expect_true(any(grepl("log_dist", output)))
  expect_true(any(grepl("cntg", output)))
})
