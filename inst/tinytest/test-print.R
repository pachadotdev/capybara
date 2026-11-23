# srr_stats (tests)
# {G1.0} Implements unit testing for print functionality.
# {G3.2} Verifies correct print output format.
# {RE4.17} Validates default print methods for models and summaries.


local({
  if (Sys.getenv("CAPYBARA_FULL_TESTING") != "yes") {
    return(NULL)
  }

  # print.feglm shows coefficients ----

  rose2004_subset <- rose2004[rose2004$year == 1999, ]
  rose2004_subset <- rose2004_subset[rose2004_subset$ltrade > quantile(rose2004_subset$ltrade, 0.75), ]

  mod <- fepoisson(ltrade ~ ldist | ctry1, rose2004_subset)

  output <- capture.output(print(mod))

  expect_true(any(grepl("ldist", output)))

  # print.felm shows coefficients ----

  mod <- felm(ltrade ~ ldist | ctry1, rose2004_subset)

  output <- capture.output(print(mod))

  expect_true(any(grepl("ldist", output)))

  # summary.feglm shows formula ----

  mod <- fepoisson(ltrade ~ ldist | ctry1, rose2004_subset)

  output <- capture.output(print(summary(mod)))

  expect_true(any(grepl("Formula", output)))

  # summary.feglm shows family ----

  mod <- fepoisson(ltrade ~ ldist | ctry1, rose2004_subset)

  output <- capture.output(print(summary(mod)))

  expect_true(any(
    grepl("Family", output) | grepl("poisson", output, ignore.case = TRUE)
  ))

  # summary.feglm shows estimates ----

  mod <- fepoisson(ltrade ~ ldist | ctry1, rose2004_subset)

  output <- capture.output(print(summary(mod)))

  expect_true(any(grepl("Estimate", output)))

  # summary.feglm shows significance codes ----

  mod <- fepoisson(ltrade ~ ldist | ctry1, rose2004_subset)

  output <- capture.output(print(summary(mod)))

  expect_true(any(grepl("Significance", output)))

  # summary.felm shows R-squared ----

  mod <- felm(ltrade ~ ldist | ctry1, rose2004_subset)

  output <- capture.output(print(summary(mod)))

  expect_true(any(grepl("R-squared", output)))

  # print shows multiple predictors ----

  mod <- felm(ltrade ~ ldist + border | ctry1, rose2004_subset)

  output <- capture.output(print(summary(mod)))

  expect_true(any(grepl("ldist", output)))
  expect_true(any(grepl("border", output)))
})
