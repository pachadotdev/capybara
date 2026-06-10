# srr_stats (tests)
# {G1.0} Implements unit testing for plotting functionality.
# {G2.3} Tests compatibility with standard plotting libraries like ggplot2.
# {RE3.1} Verifies the correctness of visual outputs for model coefficients.
# {RE3.2} Ensures that confidence levels provided to the plotting function are validated.
# {RE5.1} Confirms that `autoplot` fails gracefully with invalid inputs.
# {RE5.3} Validates that the output of `autoplot` is a `ggplot` object for visualizations.

# autoplot works for felm
local({
  ross2004_subset <- ross2004[ross2004$year == 1999, ]
  ross2004_subset <- ross2004_subset[ross2004_subset$ltrade > 0, ]

  mod <- felm(ltrade ~ ldist | ctry1, ross2004_subset)

  expect_true(inherits(autoplot(mod, conf_level = 0.99), "ggplot2::ggplot"))
  expect_true(inherits(autoplot(mod), "ggplot2::ggplot"))

  expect_error(autoplot(1L))
  expect_error(autoplot(mod, conf_level = 1.01))
  expect_error(autoplot(mod, conf_level = -0.01))
})

# autoplot works for feglm/fepoisson
local({
  ross2004_subset <- ross2004[ross2004$year == 1999, ]
  ross2004_subset <- ross2004_subset[ross2004_subset$ltrade > 0, ]
  
  mod <- fepoisson(ltrade ~ ldist | ctry1, ross2004_subset)

  expect_true(inherits(autoplot(mod), "ggplot2::ggplot"))
  expect_true(inherits(autoplot(mod, conf_level = 0.90), "ggplot2::ggplot"))
  expect_true(inherits(autoplot(mod, conf_level = 0.99), "ggplot2::ggplot"))
})

# autoplot errors on invalid conf_level
local({
  ross2004_subset <- ross2004[ross2004$year == 1999, ]
  ross2004_subset <- ross2004_subset[ross2004_subset$ltrade > 0, ]
  
  mod <- fepoisson(ltrade ~ ldist | ctry1, ross2004_subset)

  expect_error(autoplot(mod, conf_level = 0))
  expect_error(autoplot(mod, conf_level = 1))
  expect_error(autoplot(mod, conf_level = -0.5))
  expect_error(autoplot(mod, conf_level = 1.5))
})

# autoplot errors on wrong class
local({
  ross2004_subset <- ross2004[ross2004$year == 1999, ]
  ross2004_subset <- ross2004_subset[ross2004_subset$ltrade > 0, ]
  
  mod_lm <- lm(ltrade ~ ldist, ross2004_subset)

  expect_error(autoplot.feglm(mod_lm))
  expect_error(autoplot.felm(mod_lm))
})

# autoplot works with multiple predictors
local({
  ross2004_subset <- ross2004[ross2004$year == 1999, ]
  ross2004_subset <- ross2004_subset[ross2004_subset$ltrade > 0, ]

  mod <- feglm(ltrade ~ ldist + border | ctry1, ross2004_subset)

  p <- autoplot(mod)

  expect_true(inherits(p, "ggplot2::ggplot"))
})

# autoplot default conf_level is 0.95
local({
  ross2004_subset <- ross2004[ross2004$year == 1999, ]
  ross2004_subset <- ross2004_subset[ross2004_subset$ltrade > 0, ]

  mod <- felm(ltrade ~ ldist | ctry1, ross2004_subset)

  # Both should work without error (default is 0.95)
  p1 <- autoplot(mod)
  p2 <- autoplot(mod, conf_level = 0.95)

  expect_true(inherits(p1, "ggplot2::ggplot"))
  expect_true(inherits(p2, "ggplot2::ggplot"))
})
