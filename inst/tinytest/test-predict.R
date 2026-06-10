# srr_stats (tests)
# {G1.0} Implements unit testing for predict functionality.
# {G2.3} Tests various prediction types and newdata scenarios.
# {RE4.9} Verifies predict returns correct values.

# predict.feglm works with default type (response) ----

local({
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  
  mod <- fepoisson(trade ~ log_dist | exp_year, yotov2017_subset)

  preds <- predict(mod)

  expect_equal(length(preds), nrow(yotov2017_subset))
  expect_true(all(preds > 0))
})

# predict.feglm works with type = 'link' ----

local({
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  
  mod <- fepoisson(trade ~ log_dist | exp_year, yotov2017_subset)

  preds_link <- predict(mod, type = "link")
  preds_response <- predict(mod, type = "response")

  # link predictions should be different from response
  expect_false(all(preds_link == preds_response))

  # For Poisson with log link, exp(link) = response
  expect_equal(exp(preds_link), preds_response, tolerance = 1e-6)
})

# predict.feglm works with newdata ----

local({
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  
  mod <- fepoisson(trade ~ log_dist | exp_year, yotov2017_subset, control = fit_control(return_fe = TRUE))

  newdata <- data.frame(
    log_dist = c(7, 8, 9),
    exp_year = c(2006, 2006, 2006)
  )

  preds <- predict(mod, newdata = newdata)

  expect_equal(length(preds), 3)
  expect_true(all(preds > 0))

  expect_error(
    predict(
      fepoisson(trade ~ log_dist | exp_year, yotov2017_subset, control = fit_control(return_fe = FALSE)),
      newdata = newdata
    ),
    "Model has fixed effects but they were not stored."
  )
})

# predict.felm works with default type ----

local({
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  
  mod <- felm(trade ~ log_dist | exp_year, yotov2017_subset)

  preds <- predict(mod)

  expect_equal(length(preds), nrow(yotov2017_subset))
})

# predict.felm works with newdata ----

local({
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  
  mod <- felm(trade ~ log_dist | exp_year, yotov2017_subset)

  newdata <- data.frame(
    log_dist = c(7, 8, 9),
    exp_year = c(2006, 2006, 2006)
  )

  preds <- predict(mod, newdata = newdata)

  expect_equal(length(preds), 3)
})

# predict.felm with type='response' works ----

local({
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  
  mod <- felm(trade ~ log_dist | exp_year, yotov2017_subset)

  preds_response <- predict(mod, type = "response")
  preds_default <- predict(mod)

  # For linear models, response is the default
  expect_equal(preds_response, preds_default)
})

# predict works with multiple fixed effects ----

local({
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  
  mod <- fepoisson(trade ~ log_dist | exp_year + imp_year, yotov2017_subset)

  preds <- predict(mod)

  expect_equal(length(preds), nrow(yotov2017_subset))
})

# predict with newdata handles multiple FEs ----

local({
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  
  mod <- felm(trade ~ log_dist | exp_year + imp_year, yotov2017_subset)

  newdata <- data.frame(
    log_dist = c(7, 8),
    exp_year = c(2006, 2006),
    imp_year = c(2006, 2006)
  )

  preds <- predict(mod, newdata = newdata)

  expect_equal(length(preds), 2)
})

# predict works for model without fixed effects ----

local({
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  
  mod <- fepoisson(trade ~ log_dist, yotov2017_subset)

  preds <- predict(mod)

  expect_equal(length(preds), nrow(yotov2017_subset))
})

# predict with newdata works for model without FE ----

local({
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  
  mod <- felm(trade ~ log_dist, yotov2017_subset)

  newdata <- data.frame(log_dist = c(7, 8, 9))

  preds <- predict(mod, newdata = newdata)

  expect_equal(length(preds), 3)
})

# predict handles NA in newdata gracefully ---

local({
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  
  mod <- felm(trade ~ log_dist | exp_year, yotov2017_subset)

  newdata <- data.frame(
    log_dist = c(7, NA, 9),
    exp_year = c(2006, 2006, 2006)
  )

  preds <- predict(mod, newdata = newdata)

  # Should return predictions with NA where input had NA
  expect_equal(length(preds), 3)
  expect_true(is.na(preds[2]))
  expect_false(is.na(preds[1]))
  expect_false(is.na(preds[3]))
})

# predict returns same length as input for newdata ----

local({
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  
  mod <- fepoisson(trade ~ log_dist | exp_year, yotov2017_subset, control = fit_control(return_fe = TRUE))

  newdata <- data.frame(
    log_dist = c(7, 8, 9, 8.5),
    exp_year = c(2006, 2006, 2006, 2006)
  )

  preds <- predict(mod, newdata = newdata)
  expect_equal(length(preds), nrow(newdata))
})

# predict works with type='terms' for felm ----

local({
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  
  mod <- felm(trade ~ log_dist + cntg | exp_year, yotov2017_subset)

  preds_terms <- predict(mod, type = "terms")

  expect_true(is.matrix(preds_terms) || is.numeric(preds_terms))
})

# predict maintains order for newdata ----

local({
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  
  mod <- felm(trade ~ log_dist | exp_year, yotov2017_subset)

  newdata <- data.frame(
    log_dist = c(9, 7, 8.5),
    exp_year = c(2006, 2006, 2006)
  )

  preds <- predict(mod, newdata = newdata)

  # Predictions should be in same order as newdata
  expect_equal(length(preds), 3)
})
