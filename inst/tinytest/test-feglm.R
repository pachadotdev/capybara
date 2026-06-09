#' srr_stats (tests)
#' @srrstats {G5.2} Confirms that prediction errors increase outside the inter-quartile range, ensuring model generalization testing.
#' @srrstats {RE2.1} Ensures that models throw meaningful error messages when input parameters or data are invalid.
#' @srrstats {RE3.2} Compares model outputs (coefficients and fixed effects) against established benchmarks like base R's `glm`.
#' @srrstats {RE3.3} Confirms consistency of fixed effects and structural parameters between `feglm` and equivalent base models.
#' @srrstats {RE4.3} Tests robustness of predicted values using inter-quartile and outlier data subsets.
#' @srrstats {RE4.15} This is not a time-series package, so I show that the error increases when we predict outside the inter-quartile range.
#' @srrstats {RE5.1} Validates appropriate error handling for omitted arguments, such as missing formula or data.
#' @srrstats {RE5.2} Confirms that incorrect control settings result in appropriate error messages.
#' @srrstats {RE5.3} Verifies that the function stops execution when given unsupported model families or inappropriate responses.
#' @srrstats {RE5.4} Ensures that the model gracefully handles invalid starting values for beta, eta, or theta.
#' @srrstats {RE5.5} Ensures accuracy of prediction methods with unseen data subsets, maintaining expected patterns of error.
#' @srrstats {RE6.0} Implements robust testing for invalid combinations of fixed effects or missing parameters in APEs and GLMs.
#' @srrstats {RE7.1} Validates consistency in output types and structures across all supported families and link functions.
#' @srrstats {RE7.2} Confirms that confidence intervals and standard errors are computed correctly for coefficients.
#' @noRd
NULL

source(system.file("tinytest", "helper.R", package = "capybara"))

# feglm works without fixed effects
local({
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  m1 <- feglm(trade ~ log_dist, data = yotov2017_subset)
  m2 <- glm(trade ~ log_dist, data = yotov2017_subset)

  expect_equal(coef(m1), coef(m2), tolerance = 1e-6)
})

# out of sample predictions have larger margins of error
local({
  # Helper function for MAPE calculation
  mape <- function(y, yhat) {
    mean(abs(y - yhat) / y)
  }

  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017[yotov2017$trade > 0, ]

  yotov2017_subset2 <- yotov2017_subset[
    yotov2017_subset$trade >= quantile(yotov2017_subset$trade, 0.25) &
      yotov2017_subset$trade <= quantile(yotov2017_subset$trade, 0.75),
  ]

  mod <- fepoisson(trade ~ log_dist | exp_year, yotov2017_subset2)
  
  p1 <- predict(mod, newdata = yotov2017_subset, type = "response")
  p2 <- predict(mod, newdata = yotov2017_subset2, type = "response")

  mape1 <- mape(yotov2017_subset$trade, p1)
  mape2 <- mape(yotov2017_subset2$trade, p2)

  expect_true(mape1 > mape2)
})

# proportional regressors return NA coefficients
local({
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
})

# feglm with weights works
local({
  skip_on_cran()

  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset$trade_pair <- ave(yotov2017_subset$trade, yotov2017_subset$pair,
    FUN = function(x) sum(x, na.rm = TRUE))

  m1 <- feglm(trade ~ log_dist | exp_year, weights = ~trade_pair, data = yotov2017_subset)
  m2 <- feglm(trade ~ log_dist | exp_year, weights = yotov2017_subset$trade_pair, data = yotov2017_subset)

  w <- yotov2017_subset$trade_pair
  m3 <- feglm(trade ~ log_dist | exp_year, weights = w, data = yotov2017_subset)

  expect_equal(coef(m2), coef(m1))
  expect_equal(coef(m3), coef(m1))

  w <- NULL
  m4 <- feglm(trade ~ log_dist | exp_year, weights = w, data = yotov2017_subset)

  expect_true(coef(m1) != coef(m4))
})

# feglm works without fixed effects (stammann centering)
local({
  # centering is unused without FEs, but control must be accepted
  ctrl <- list(centering = "stammann")

  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]

  m1 <- feglm(trade ~ log_dist, data = yotov2017_subset, control = ctrl)
  m2 <- glm(trade ~ log_dist, data = yotov2017_subset)

  expect_equal(coef(m1), coef(m2), tolerance = 1e-6)
})

# proportional regressors return NA coefficients (stammann centering)
local({
  ctrl <- list(centering = "stammann")
  set.seed(200100)
  d <- data.frame(
    y = rnorm(100),
    x1 = rnorm(100),
    f = factor(sample(1:2, 100, replace = TRUE))
  )
  d$x2 <- 2 * d$x1

  fit1 <- glm(y ~ x1 + x2 + as.factor(f), data = d, family = gaussian())
  fit2 <- feglm(y ~ x1 + x2 | f, data = d, family = gaussian(), control = ctrl)

  expect_equal(coef(fit2), coef(fit1)[2:3], tolerance = 1e-2)
  expect_equal(predict(fit2), predict(fit1), tolerance = 1e-2)
})
