# srr_stats (tests)
# {G5.2} Confirms that prediction errors increase outside the inter-quartile range, ensuring model generalization testing.
# {RE2.1} Ensures that models throw meaningful error messages when input parameters or data are invalid.
# {RE3.2} Compares model outputs (coefficients and fixed effects) against established benchmarks like base R's `glm`.
# {RE3.3} Confirms consistency of fixed effects and structural parameters between `feglm` and equivalent base models.
# {RE4.3} Tests robustness of predicted values using inter-quartile and outlier data subsets.
# {RE4.15} This is not a time-series package, so I show that the error increases when we predict outside the inter-quartile range.
# {RE5.1} Validates appropriate error handling for omitted arguments, such as missing formula or data.
# {RE5.2} Confirms that incorrect control settings result in appropriate error messages.
# {RE5.3} Verifies that the function stops execution when given unsupported model families or inappropriate responses.
# {RE5.4} Ensures that the model gracefully handles invalid starting values for beta, eta, or theta.
# {RE5.5} Ensures accuracy of prediction methods with unseen data subsets, maintaining expected patterns of error.
# {RE6.0} Implements robust testing for invalid combinations of fixed effects or missing parameters in APEs and GLMs.
# {RE7.1} Validates consistency in output types and structures across all supported families and link functions.
# {RE7.2} Confirms that confidence intervals and standard errors are computed correctly for coefficients.

source(system.file("tinytest", "helper.R", package = "capybara"))

local({
  # feglm works without fixed effects ----

  ross2004_subset <- ross2004[ross2004$year == 1999, ]
  ross2004_subset <- ross2004_subset[ross2004_subset$ltrade > quantile(ross2004_subset$ltrade, 0.75), ]

  m1 <- feglm(ltrade ~ ldist, data = ross2004_subset)
  m2 <- glm(ltrade ~ ldist, data = ross2004_subset)

  expect_equal(coef(m1), coef(m2), tolerance = 1e-6)

  # out of sample predictions have larger margins of error ----

  ross2004_subset2 <- ross2004_subset[
    ross2004_subset$ltrade >= quantile(ross2004_subset$ltrade, 0.25) &
      ross2004_subset$ltrade <= quantile(ross2004_subset$ltrade, 0.75),
  ]

  mod <- fepoisson(ltrade ~ ldist | ctry1, ross2004_subset2)

  p1 <- predict(mod, newdata = ross2004_subset, type = "response")
  p2 <- predict(mod, newdata = ross2004_subset2, type = "response")

  mape1 <- mape(ross2004_subset$ltrade, p1)
  mape2 <- mape(ross2004_subset2$ltrade, p2)

  expect_true(mape1 > mape2)

  # proportional regressors return NA coefficients ----

  set.seed(200100)
  d <- data.frame(
    y = rnorm(100),
    x1 = rnorm(100),
    f = factor(sample(1:2, 100, replace = TRUE)) # Fixed: was 1000, now 100
  )
  d$x2 <- 2 * d$x1

  fit1 <- glm(y ~ x1 + x2 + as.factor(f), data = d, family = gaussian())
  fit2 <- feglm(y ~ x1 + x2 | f, data = d, family = gaussian())

  expect_equal(coef(fit2), coef(fit1)[2:3], tolerance = 1e-2)
  expect_equal(predict(fit2), predict(fit1), tolerance = 1e-2)

  # feglm with weights works ----

  ross2004_subset <- ross2004[ross2004$year == 1999, ]
  ross2004_subset$trade_pair <- ave(ross2004_subset$ltrade, ross2004_subset$pair,
    FUN = function(x) sum(x, na.rm = TRUE)
  )

  m1 <- feglm(ltrade ~ ldist | ctry1, weights = ~trade_pair, data = ross2004_subset)
  m2 <- feglm(ltrade ~ ldist | ctry1, weights = ross2004_subset$trade_pair, data = ross2004_subset)

  w <- ross2004_subset$trade_pair
  m3 <- feglm(ltrade ~ ldist | ctry1, weights = w, data = ross2004_subset)

  expect_equal(coef(m2), coef(m1))
  expect_equal(coef(m3), coef(m1))

  w <- NULL
  m4 <- feglm(ltrade ~ ldist | ctry1, weights = w, data = ross2004_subset)

  expect_true(coef(m1) != coef(m4))
})
