# srr_stats (tests)
# {G1.0} Implements unit testing for plotting functionality.
# {G2.3} Tests compatibility with standard plotting libraries like ggplot2.
# {RE3.1} Verifies the correctness of visual outputs for model coefficients.
# {RE3.2} Ensures that confidence levels provided to the plotting function are validated.
# {RE5.1} Confirms that `autoplot` fails gracefully with invalid inputs.
# {RE5.3} Validates that the output of `autoplot` is a `ggplot` object for visualizations.

local({
  if (Sys.getenv("CAPYBARA_FULL_TESTING") != "yes") {
    return(NULL)
  }

  # autoplot works for felm ----

  ross2004_subset <- ross2004[ross2004$year == 1999, ]
  ross2004_subset <- ross2004_subset[ross2004_subset$ltrade > quantile(ross2004_subset$ltrade, 0.75), ]

  mod <- felm(ltrade ~ ldist | ctry1, ross2004_subset)

  expect_true(inherits(autoplot(mod, conf_level = 0.99), "ggplot2::ggplot"))
  expect_true(inherits(autoplot(mod), "ggplot2::ggplot"))

  expect_error(autoplot(1L))
  expect_error(autoplot(mod, conf_level = 1.01))
  expect_error(autoplot(mod, conf_level = -0.01))

  # autoplot works for feglm/fepoisson ----

  mod <- fepoisson(ltrade ~ ldist | ctry1, ross2004_subset)

  expect_true(inherits(autoplot(mod), "ggplot2::ggplot"))
  expect_true(inherits(autoplot(mod, conf_level = 0.90), "ggplot2::ggplot"))

  # autoplot errors on invalid conf_level ----

  mod <- fepoisson(ltrade ~ ldist | ctry1, ross2004_subset)

  expect_error(autoplot(mod, conf_level = 0))
  expect_error(autoplot(mod, conf_level = 1))
  expect_error(autoplot(mod, conf_level = -0.5))
  expect_error(autoplot(mod, conf_level = 1.5))

  # autoplot errors on wrong class ----

  mod_lm <- lm(ltrade ~ ldist, ross2004_subset)

  expect_error(autoplot.feglm(mod_lm))
  expect_error(autoplot.felm(mod_lm))
})
