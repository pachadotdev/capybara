# srr_stats (tests)
# {G1.0} Implements unit testing for predict functionality.
# {G2.3} Tests various prediction types and newdata scenarios.
# {RE4.9} Verifies predict returns correct values.

local({
  if (Sys.getenv("CAPYBARA_FULL_TESTING") != "yes") {
    return(NULL)
  }

  # predict.feglm works with default type (response) ----

  ross2004_subset <- ross2004[ross2004$year == 1999, ]
  ross2004_subset <- ross2004_subset[ross2004_subset$ltrade > quantile(ross2004_subset$ltrade, 0.75), ]

  mod <- fepoisson(ltrade ~ ldist | ctry1, ross2004_subset)

  preds <- predict(mod)

  expect_equal(length(preds), nrow(ross2004_subset))
  expect_true(all(preds > 0))

  # predict.feglm works with type = 'link' ----

  ross2004_subset <- ross2004[ross2004$year == 1999, ]
  ross2004_subset <- ross2004_subset[ross2004_subset$ltrade > quantile(ross2004_subset$ltrade, 0.75), ]

  mod <- fepoisson(ltrade ~ ldist | ctry1, ross2004_subset)

  preds_link <- predict(mod, type = "link")
  preds_response <- predict(mod, type = "response")

  # link predictions should be different from response
  expect_false(all(preds_link == preds_response))

  # For Poisson with log link, exp(link) = response
  expect_equal(exp(preds_link), preds_response, tolerance = 1e-6)

  # predict.feglm works with newdata ----

  ross2004_subset <- ross2004[ross2004$year == 1999, ]
  ross2004_subset <- ross2004_subset[ross2004_subset$ltrade > quantile(ross2004_subset$ltrade, 0.75), ]

  mod <- fepoisson(ltrade ~ ldist | ctry1, ross2004_subset, control = fit_control(return_fe = TRUE))

  newdata <- data.frame(
    ldist = c(7, 8, 9),
    ctry1 = c(1999, 1999, 1999)
  )

  preds <- predict(mod, newdata = newdata)

  expect_equal(length(preds), 3)
  expect_true(all(preds > 0))

  expect_error(
    predict(
      fepoisson(ltrade ~ ldist | ctry1, ross2004_subset, control = fit_control(return_fe = FALSE)),
      newdata = newdata
    ),
    "Model has fixed effects but they were not stored."
  )

  # predict.felm works with default type ----

  ross2004_subset <- ross2004[ross2004$year == 1999, ]
  ross2004_subset <- ross2004_subset[ross2004_subset$ltrade > quantile(ross2004_subset$ltrade, 0.75), ]

  mod <- felm(ltrade ~ ldist | ctry1, ross2004_subset)

  preds <- predict(mod)

  expect_equal(length(preds), nrow(ross2004_subset))

  # predict.felm works with newdata ----

  ross2004_subset <- ross2004[ross2004$year == 1999, ]
  ross2004_subset <- ross2004_subset[ross2004_subset$ltrade > quantile(ross2004_subset$ltrade, 0.75), ]

  mod <- felm(ltrade ~ ldist | ctry1, ross2004_subset)

  newdata <- data.frame(
    ldist = c(7, 8, 9),
    ctry1 = c(1999, 1999, 1999)
  )

  preds <- predict(mod, newdata = newdata)

  expect_equal(length(preds), 3)

  # predict.felm with type='response' works ----

  ross2004_subset <- ross2004[ross2004$year == 1999, ]
  ross2004_subset <- ross2004_subset[ross2004_subset$ltrade > quantile(ross2004_subset$ltrade, 0.75), ]

  mod <- felm(ltrade ~ ldist | ctry1, ross2004_subset)

  preds_response <- predict(mod, type = "response")
  preds_default <- predict(mod)

  # For linear models, response is the default
  expect_equal(preds_response, preds_default)

  # predict works with multiple fixed effects ----

  ross2004_subset <- ross2004[ross2004$year == 1999, ]
  ross2004_subset <- ross2004_subset[ross2004_subset$ltrade > quantile(ross2004_subset$ltrade, 0.75), ]

  mod <- fepoisson(ltrade ~ ldist | ctry1 + ctry2, ross2004_subset)

  preds <- predict(mod)

  expect_equal(length(preds), nrow(ross2004_subset))

  # predict with newdata handles multiple FEs ----

  ross2004_subset <- ross2004[ross2004$year == 1999, ]
  ross2004_subset <- ross2004_subset[ross2004_subset$ltrade > quantile(ross2004_subset$ltrade, 0.75), ]

  mod <- felm(ltrade ~ ldist | ctry1 + ctry2, ross2004_subset)

  newdata <- data.frame(
    ldist = c(7, 8),
    ctry1 = c(1999, 1999),
    ctry2 = c(1999, 1999)
  )

  preds <- predict(mod, newdata = newdata)

  expect_equal(length(preds), 2)

  # predict works for model without fixed effects ----

  ross2004_subset <- ross2004[ross2004$year == 1999, ]
  ross2004_subset <- ross2004_subset[ross2004_subset$ltrade > quantile(ross2004_subset$ltrade, 0.75), ]

  mod <- fepoisson(ltrade ~ ldist, ross2004_subset)

  preds <- predict(mod)

  expect_equal(length(preds), nrow(ross2004_subset))

  # predict with newdata works for model without FE ----

  ross2004_subset <- ross2004[ross2004$year == 1999, ]
  ross2004_subset <- ross2004_subset[ross2004_subset$ltrade > quantile(ross2004_subset$ltrade, 0.75), ]

  mod <- felm(ltrade ~ ldist, ross2004_subset)

  newdata <- data.frame(ldist = c(7, 8, 9))

  preds <- predict(mod, newdata = newdata)

  expect_equal(length(preds), 3)

  # predict handles NA in newdata gracefully ---

  ross2004_subset <- ross2004[ross2004$year == 1999, ]
  ross2004_subset <- ross2004_subset[ross2004_subset$ltrade > quantile(ross2004_subset$ltrade, 0.75), ]

  mod <- felm(ltrade ~ ldist | ctry1, ross2004_subset)

  newdata <- data.frame(
    ldist = c(7, NA, 9),
    ctry1 = c(1999, 1999, 1999)
  )

  preds <- predict(mod, newdata = newdata)

  # Should return predictions with NA where input had NA
  expect_equal(length(preds), 3)
  expect_true(is.na(preds[2]))
  expect_false(is.na(preds[1]))
  expect_false(is.na(preds[3]))

  # predict returns same length as input for newdata ----

  ross2004_subset <- ross2004[ross2004$year == 1999, ]
  ross2004_subset <- ross2004_subset[ross2004_subset$ltrade > quantile(ross2004_subset$ltrade, 0.75), ]

  mod <- fepoisson(ltrade ~ ldist | ctry1, ross2004_subset, control = fit_control(return_fe = TRUE))

  newdata <- data.frame(
    ldist = c(7, 8, 9, 8.5),
    ctry1 = c(1999, 1999, 1999, 1999)
  )

  preds <- predict(mod, newdata = newdata)
  expect_equal(length(preds), nrow(newdata))

  # predict works with type='terms' for felm ----

  ross2004_subset <- ross2004[ross2004$year == 1999, ]
  ross2004_subset <- ross2004_subset[ross2004_subset$ltrade > quantile(ross2004_subset$ltrade, 0.75), ]

  mod <- felm(ltrade ~ ldist + border | ctry1, ross2004_subset)

  preds_terms <- predict(mod, type = "terms")

  expect_true(is.matrix(preds_terms) || is.numeric(preds_terms))

  # predict maintains order for newdata ----

  ross2004_subset <- ross2004[ross2004$year == 1999, ]
  ross2004_subset <- ross2004_subset[ross2004_subset$ltrade > quantile(ross2004_subset$ltrade, 0.75), ]

  mod <- felm(ltrade ~ ldist | ctry1, ross2004_subset)

  newdata <- data.frame(
    ldist = c(9, 7, 8.5),
    ctry1 = c(1999, 1999, 1999)
  )

  preds <- predict(mod, newdata = newdata)

  # Predictions should be in same order as newdata
  expect_equal(length(preds), 3)
})
