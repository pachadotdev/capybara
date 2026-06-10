# srr_stats (tests)
# {G1.0} Implements unit testing for print functionality.
# {G3.2} Verifies correct print output format.
# {RE4.17} Validates default print methods for models and summaries.

# ---- print.feglm tests ----

# print.feglm shows coefficients
local({
  ross2004_subset <- ross2004[ross2004$year == 1999, ]
  ross2004_subset <- ross2004_subset[ross2004_subset$ltrade > 0, ]
  
  mod <- fepoisson(ltrade ~ ldist | ctry1, ross2004_subset)

  output <- capture.output(print(mod))

  expect_true(any(grepl("ldist", output)))
})

# ---- print.felm tests ----

# print.felm shows coefficients
local({
  ross2004_subset <- ross2004[ross2004$year == 1999, ]
  ross2004_subset <- ross2004_subset[ross2004_subset$ltrade > 0, ]
  
  mod <- felm(ltrade ~ ldist | ctry1, ross2004_subset)

  output <- capture.output(print(mod))

  expect_true(any(grepl("ldist", output)))
})

# ---- print.summary.feglm tests ----

# summary.feglm shows formula
local({
  ross2004_subset <- ross2004[ross2004$year == 1999, ]
  ross2004_subset <- ross2004_subset[ross2004_subset$ltrade > 0, ]
  
  mod <- fepoisson(ltrade ~ ldist | ctry1, ross2004_subset)

  output <- capture.output(print(summary(mod)))

  expect_true(any(grepl("Formula", output)))
})

# summary.feglm shows family
local({
  ross2004_subset <- ross2004[ross2004$year == 1999, ]
  ross2004_subset <- ross2004_subset[ross2004_subset$ltrade > 0, ]
  
  mod <- fepoisson(ltrade ~ ldist | ctry1, ross2004_subset)

  output <- capture.output(print(summary(mod)))

  expect_true(any(
    grepl("Family", output) | grepl("poisson", output, ignore.case = TRUE)
  ))
})

# summary.feglm shows estimates
local({
  ross2004_subset <- ross2004[ross2004$year == 1999, ]
  ross2004_subset <- ross2004_subset[ross2004_subset$ltrade > 0, ]
  
  mod <- fepoisson(ltrade ~ ldist | ctry1, ross2004_subset)

  output <- capture.output(print(summary(mod)))

  expect_true(any(grepl("Estimate", output)))
})

# summary.feglm shows significance codes
local({
  ross2004_subset <- ross2004[ross2004$year == 1999, ]
  ross2004_subset <- ross2004_subset[ross2004_subset$ltrade > 0, ]
  
  mod <- fepoisson(ltrade ~ ldist | ctry1, ross2004_subset)

  output <- capture.output(print(summary(mod)))

  expect_true(any(grepl("Significance", output)))
})

# ---- print.summary.felm tests ----

# summary.felm shows R-squared
local({
  ross2004_subset <- ross2004[ross2004$year == 1999, ]
  ross2004_subset <- ross2004_subset[ross2004_subset$ltrade > 0, ]
  
  mod <- felm(ltrade ~ ldist | ctry1, ross2004_subset)

  output <- capture.output(print(summary(mod)))

  expect_true(any(grepl("R-squared|R²", output)))
})

# ---- print with multiple predictors ----

# print shows multiple predictors
local({
  ross2004_subset <- ross2004[ross2004$year == 1999, ]
  ross2004_subset <- ross2004_subset[ross2004_subset$ltrade > 0, ]
  
  mod <- felm(ltrade ~ ldist + border | ctry1, ross2004_subset)

  output <- capture.output(print(summary(mod)))

  expect_true(any(grepl("ldist", output)))
  expect_true(any(grepl("border", output)))
})
