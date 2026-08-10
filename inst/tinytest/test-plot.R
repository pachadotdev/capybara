# srr_stats (tests)
# {G1.0} Implements unit testing for plotting functionality.
# {G2.3} Tests compatibility with standard plotting libraries like tinyplot.
# {RE3.1} Verifies the correctness of visual outputs for model coefficients.
# {RE3.2} Ensures that confidence levels provided to the plotting function are validated.
# {RE5.1} Confirms that `plot` fails gracefully with invalid inputs.
# {RE5.3} Validates that `plot` produces a plot without erroring.

local({
  if (Sys.getenv("CAPYBARA_FULL_TESTING") != "yes") {
    return(NULL)
  }

  # plot works for felm ----

  ross2004_subset <- ross2004[ross2004$year == 1999, ]
  ross2004_subset <- ross2004_subset[ross2004_subset$ltrade > quantile(ross2004_subset$ltrade, 0.75), ]

  mod <- felm(ltrade ~ ldist | ctry1, ross2004_subset)

  expect_silent(plot(mod, conf_level = 0.99))
  expect_silent(plot(mod))

  expect_error(plot.felm(mod, conf_level = 1.01))
  expect_error(plot.felm(mod, conf_level = -0.01))

  # plot works for feglm/fepoisson ----

  mod <- fepoisson(ltrade ~ ldist | ctry1, ross2004_subset)

  expect_silent(plot(mod))
  expect_silent(plot(mod, conf_level = 0.90))

  # plot errors on invalid conf_level ----

  mod <- fepoisson(ltrade ~ ldist | ctry1, ross2004_subset)

  expect_error(plot.feglm(mod, conf_level = 0))
  expect_error(plot.feglm(mod, conf_level = 1))
  expect_error(plot.feglm(mod, conf_level = -0.5))
  expect_error(plot.feglm(mod, conf_level = 1.5))

  # plot errors on wrong class ----

  mod_lm <- lm(ltrade ~ ldist, ross2004_subset)

  expect_error(plot.feglm(mod_lm))
  expect_error(plot.felm(mod_lm))
})
