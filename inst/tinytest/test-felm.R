#' srr_stats (tests)
#' @srrstats {RE2.1} Ensures that models throw meaningful error messages when input parameters or data are invalid.
#' @srrstats {RE3.1} Validates consistency between `felm` and base R `lm` in terms of coefficients, R-squared, and fitted values.
#' @srrstats {RE3.2} Compares model outputs against established benchmarks such as base R's `lm`.
#' @srrstats {RE5.1} Validates appropriate error handling for omitted arguments or missing data.
#' @srrstats {RE6.0} Implements robust testing for invalid or collinear regressors.
#' @srrstats {RE7.1} Validates that proportional regressors or collinear terms are detected and produce errors.
#' @srrstats {RE7.1a} Adding noise to the depending variable minimally affects the speed. I tested that explicitly.
#' @srrstats {RE7.2} Confirms that model computations remain consistent when small noise is added to data.
#' @srrstats {RE8.1} Ensures computational times remain consistent under similar model specifications.
#' @noRd
NULL

source(system.file("tinytest", "helper.R", package = "capybara"))

# felm works
local({
  # Setup yotov2017 data
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  
  # 1-FE ----

  m1 <- felm(formula = trade ~ log_dist | exp_year, data = yotov2017_subset)
  m2 <- lm(trade ~ log_dist + as.factor(exp_year), yotov2017_subset)

  expect_equal(coef(m1), coef(m2)[2], tolerance = 1e-2)

  n <- nrow(yotov2017_subset)
  expect_equal(length(fitted(m1)), n)
  expect_equal(length(predict(m1)), n)
  expect_equal(length(coef(m1)), 1)
  expect_equal(length(coef(summary(m1))), 4)

  m1 <- felm(trade ~ log_dist + cntg | exp_year, yotov2017_subset)
  m2 <- lm(trade ~ log_dist + cntg + as.factor(exp_year), yotov2017_subset)

  expect_equal(coef(m1), coef(m2)[c(2, 3)], tolerance = 1e-2)

  # 2-FE ----

  m1 <- felm(trade ~ log_dist + cntg | exp_year + imp_year, yotov2017_subset)

  m2 <- lm(trade ~ log_dist + cntg + as.factor(exp_year) + as.factor(imp_year), yotov2017_subset)

  expect_equal(coef(m1), coef(m2)[c(2, 3)], tolerance = 1e-2)

  s1 <- summary(m1)
  s2 <- summary(m2)

  expect_equal(s1$r_squared, s2$r.squared, tolerance = 1e-2)
  expect_equal(s1$adj_r_squared, s2$adj.r.squared, tolerance = 1e-2)

  yotov2017_subset2 <- yotov2017_subset
  yotov2017_subset2$log_dist[2] <- NA

  m1 <- felm(trade ~ log_dist + cntg | exp_year + imp_year, yotov2017_subset2)
  m2 <- lm(trade ~ log_dist + cntg + as.factor(exp_year) + as.factor(imp_year), yotov2017_subset2)

  expect_equal(coef(m1), coef(m2)[c(2, 3)], tolerance = 1e-2)

  s1 <- summary(m1)
  s2 <- summary(m2)

  expect_equal(s1$r_squared, s2$r.squared, tolerance = 1e-2)
  expect_equal(s1$adj_r_squared, s2$adj.r.squared, tolerance = 1e-2)

  m1 <- felm(trade ~ log_dist + cntg | exp_year + imp_year | year, yotov2017_subset)

  expect_equal(coef(m1), coef(m2)[c(2, 3)], tolerance = 1e-2)

  # 3-FE ----

  m1 <- felm(trade ~ log_dist + cntg | exp_year + imp_year + year, yotov2017_subset)
  m2 <- lm(
    trade ~ log_dist + cntg + as.factor(exp_year) + as.factor(imp_year) + as.factor(year),
    yotov2017_subset
  )

  expect_equal(coef(m1), coef(m2)[c(2, 3)], tolerance = 1e-2)

  s1 <- summary(m1)
  s2 <- summary(m2)
  expect_equal(s1$r_squared, s2$r.squared, tolerance = 1e-2)
  expect_equal(s1$adj_r_squared, s2$adj.r.squared, tolerance = 1e-2)
})

# felm is correct without fixed effects
local({
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  
  m1 <- felm(trade ~ log_dist, yotov2017_subset)
  m2 <- lm(trade ~ log_dist, yotov2017_subset)

  s2 <- summary(m2)

  expect_equal(coef(m1), coef(m2), tolerance = 1e-2)

  expect_equal(m1$r_squared, s2$r.squared, tolerance = 1e-2)
  expect_equal(m1$adj_r_squared, s2$adj.r.squared, tolerance = 1e-2)
})

# felm time is the minimally affected when adding noise to the data
local({
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  
  yotov2017_subset2 <- yotov2017_subset[, c("trade", "log_dist", "exp_year")]
  set.seed(200100)
  yotov2017_subset2$trade <- yotov2017_subset2$trade + rbinom(nrow(yotov2017_subset2), 1, 0.5) * .Machine$double.eps
  m1 <- felm(trade ~ log_dist | exp_year, yotov2017_subset, control = fit_control(return_fe = TRUE))
  m2 <- felm(trade ~ log_dist | exp_year, yotov2017_subset2, control = fit_control(return_fe = TRUE))
  expect_equal(coef(m1), coef(m2))
  expect_equal(m1$fixed_effects, m2$fixed_effects)

  t1 <- rep(NA, 10)
  t2 <- rep(NA, 10)
  for (i in 1:10) {
    a <- Sys.time()
    m1 <- felm(trade ~ log_dist | exp_year, yotov2017_subset)
    b <- Sys.time()
    t1[i] <- b - a

    a <- Sys.time()
    m2 <- felm(trade ~ log_dist | exp_year, yotov2017_subset2)
    b <- Sys.time()
    t2[i] <- b - a
  }
  expect_true(abs(median(t1) - median(t2)) < 0.05)
})

# proportional regressors return NA coefficients
local({
  set.seed(200100)
  d <- data.frame(
    y = rnorm(100),
    x1 = rnorm(100),
    f = factor(sample(1:2, 100, replace = TRUE))
  )

  d$x2 <- 2 * d$x1
  fit1 <- lm(y ~ x1 + x2 + as.factor(f), data = d)
  fit2 <- felm(y ~ x1 + x2 | f, data = d)

  expect_equal(coef(fit2), coef(fit1)[2:3], tolerance = 1e-2)
  expect_equal(predict(fit2), predict(fit1), tolerance = 1e-2)
})

# felm correctly predicts values outside the inter-quartile range
local({
  # Helper function for MAPE calculation
  mape <- function(y, yhat) {
    mean(abs(y - yhat) / abs(y))
  }

  # Create data subset once
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  
  d1 <- yotov2017_subset[
    yotov2017_subset$trade >= quantile(yotov2017_subset$trade, 0.25) &
      yotov2017_subset$trade <= quantile(yotov2017_subset$trade, 0.75),
  ]
  d2 <- yotov2017_subset[
    yotov2017_subset$trade < quantile(yotov2017_subset$trade, 0.25) |
      yotov2017_subset$trade > quantile(yotov2017_subset$trade, 0.75),
  ]

  m1_lm <- felm(trade ~ log_dist + cntg | exp_year, yotov2017_subset)
  m2_lm <- lm(trade ~ log_dist + cntg + as.factor(exp_year), yotov2017_subset)

  pred1_lm <- predict(m1_lm, newdata = d1)
  pred2_lm <- predict(m1_lm, newdata = d2)

  mape1_lm <- mape(d1$trade, pred1_lm)
  mape2_lm <- mape(d2$trade, pred2_lm)

  expect_true(mape1_lm < mape2_lm)

  # Compare with base R linear model
  pred1_base_lm <- predict(m2_lm, newdata = d1)
  pred2_base_lm <- predict(m2_lm, newdata = d2)

  expect_equal(pred1_lm, pred1_base_lm, tolerance = 1e-2)
  expect_equal(pred2_lm, pred2_base_lm, tolerance = 1e-2)
})

# felm with weights works
local({
  skip_on_cran()

  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  yotov2017_subset$weight_var <- yotov2017_subset$cntg

  m1 <- felm(trade ~ log_dist | exp_year, weights = ~weight_var, data = yotov2017_subset)
  m2 <- felm(trade ~ log_dist | exp_year, weights = yotov2017_subset$weight_var, data = yotov2017_subset)

  w <- yotov2017_subset$weight_var
  m3 <- felm(trade ~ log_dist | exp_year, weights = w, data = yotov2017_subset)

  expect_equal(coef(m2), coef(m1))
  expect_equal(coef(m3), coef(m1))

  w <- NULL
  m4 <- felm(trade ~ log_dist | exp_year, weights = w, data = yotov2017_subset)

  expect_true(coef(m1) != coef(m4))
})

# Stammann centering ----

# felm works (stammann centering)
local({
  ctrl <- list(centering = "stammann")

  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  
  # 1-FE ----

  m1 <- felm(formula = trade ~ log_dist | exp_year, data = yotov2017_subset, control = ctrl)
  m2 <- lm(trade ~ log_dist + as.factor(exp_year), yotov2017_subset)

  expect_equal(coef(m1), coef(m2)[2], tolerance = 1e-2)

  n <- nrow(yotov2017_subset)
  expect_equal(length(fitted(m1)), n)
  expect_equal(length(predict(m1)), n)
  expect_equal(length(coef(m1)), 1)
  expect_equal(length(coef(summary(m1))), 4)

  m1 <- felm(trade ~ log_dist + cntg | exp_year, yotov2017_subset, control = ctrl)
  m2 <- lm(trade ~ log_dist + cntg + as.factor(exp_year), yotov2017_subset)

  expect_equal(coef(m1), coef(m2)[c(2, 3)], tolerance = 1e-2)

  # 2-FE ----

  m1 <- felm(trade ~ log_dist + cntg | exp_year + imp_year, yotov2017_subset, control = ctrl)
  m2 <- lm(trade ~ log_dist + cntg + as.factor(exp_year) + as.factor(imp_year), yotov2017_subset)

  expect_equal(coef(m1), coef(m2)[c(2, 3)], tolerance = 1e-2)

  s1 <- summary(m1)
  s2 <- summary(m2)

  expect_equal(s1$r_squared, s2$r.squared, tolerance = 1e-2)
  expect_equal(s1$adj_r_squared, s2$adj.r.squared, tolerance = 1e-2)

  yotov2017_subset2 <- yotov2017_subset
  yotov2017_subset2$log_dist[2] <- NA

  m1 <- felm(trade ~ log_dist + cntg | exp_year + imp_year, yotov2017_subset2, control = ctrl)
  m2 <- lm(trade ~ log_dist + cntg + as.factor(exp_year) + as.factor(imp_year), yotov2017_subset2)

  expect_equal(coef(m1), coef(m2)[c(2, 3)], tolerance = 1e-2)

  s1 <- summary(m1)
  s2 <- summary(m2)

  expect_equal(s1$r_squared, s2$r.squared, tolerance = 1e-2)
  expect_equal(s1$adj_r_squared, s2$adj.r.squared, tolerance = 1e-2)

  m1 <- felm(trade ~ log_dist + cntg | exp_year + imp_year | year, yotov2017_subset, control = ctrl)

  expect_equal(coef(m1), coef(m2)[c(2, 3)], tolerance = 1e-2)

  # 3-FE ----

  m1 <- felm(trade ~ log_dist + cntg | exp_year + imp_year + year, yotov2017_subset, control = ctrl)
  m2 <- lm(
    trade ~ log_dist + cntg + as.factor(exp_year) + as.factor(imp_year) + as.factor(year),
    yotov2017_subset
  )

  expect_equal(coef(m1), coef(m2)[c(2, 3)], tolerance = 1e-2)

  s1 <- summary(m1)
  s2 <- summary(m2)
  expect_equal(s1$r_squared, s2$r.squared, tolerance = 1e-2)
  expect_equal(s1$adj_r_squared, s2$adj.r.squared, tolerance = 1e-2)
})

# felm is correct without fixed effects (stammann centering)
local({
  # centering is unused when there are no FEs, but the control arg
  # must still be accepted without error
  ctrl <- list(centering = "stammann")
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  
  m1 <- felm(trade ~ log_dist, yotov2017_subset, control = ctrl)
  m2 <- lm(trade ~ log_dist, yotov2017_subset)

  s2 <- summary(m2)

  expect_equal(coef(m1), coef(m2), tolerance = 1e-2)
  expect_equal(m1$r_squared, s2$r.squared, tolerance = 1e-2)
  expect_equal(m1$adj_r_squared, s2$adj.r.squared, tolerance = 1e-2)
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
  fit1 <- lm(y ~ x1 + x2 + as.factor(f), data = d)
  fit2 <- felm(y ~ x1 + x2 | f, data = d, control = ctrl)

  expect_equal(coef(fit2), coef(fit1)[2:3], tolerance = 1e-2)
  expect_equal(predict(fit2), predict(fit1), tolerance = 1e-2)
})

# felm correctly predicts values outside the inter-quartile range (stammann centering)
local({
  ctrl <- list(centering = "stammann")

  mape <- function(y, yhat) mean(abs(y - yhat) / abs(y))

  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  
  d1 <- yotov2017_subset[
    yotov2017_subset$trade >= quantile(yotov2017_subset$trade, 0.25) &
      yotov2017_subset$trade <= quantile(yotov2017_subset$trade, 0.75),
  ]
  d2 <- yotov2017_subset[
    yotov2017_subset$trade < quantile(yotov2017_subset$trade, 0.25) |
      yotov2017_subset$trade > quantile(yotov2017_subset$trade, 0.75),
  ]

  m1_lm <- felm(trade ~ log_dist + cntg | exp_year, yotov2017_subset, control = ctrl)
  m2_lm <- lm(trade ~ log_dist + cntg + as.factor(exp_year), yotov2017_subset)

  pred1_lm <- predict(m1_lm, newdata = d1)
  pred2_lm <- predict(m1_lm, newdata = d2)

  expect_true(mape(d1$trade, pred1_lm) < mape(d2$trade, pred2_lm))
  expect_equal(pred1_lm, predict(m2_lm, newdata = d1), tolerance = 1e-2)
  expect_equal(pred2_lm, predict(m2_lm, newdata = d2), tolerance = 1e-2)
})

# felm with weights works (stammann centering)
local({
  skip_on_cran()
  ctrl <- list(centering = "stammann")

  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  yotov2017_subset$w <- rbinom(nrow(yotov2017_subset2), 1, 0.5)
  yotov2017_subset$trade_pair <- ave(yotov2017_subset$trade, yotov2017_subset$pair,
    FUN = function(x) sum(x, na.rm = TRUE))

  m1 <- felm(trade ~ log_dist | exp_year, weights = ~trade_pair, data = yotov2017_subset, control = ctrl)
  m2 <- felm(trade ~ log_dist | exp_year, weights = yotov2017_subset$trade_pair, data = yotov2017_subset, control = ctrl)

  w <- yotov2017_subset$trade_pair
  m3 <- felm(trade ~ log_dist | exp_year, weights = w, data = yotov2017_subset, control = ctrl)

  expect_equal(coef(m2), coef(m1))
  expect_equal(coef(m3), coef(m1))

  w <- NULL
  m4 <- felm(trade ~ log_dist | exp_year, weights = w, data = yotov2017_subset, control = ctrl)

  expect_true(coef(m1) != coef(m4))
})
