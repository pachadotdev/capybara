# srr_stats (tests)
# {RE2.1} Ensures that models throw meaningful error messages when input parameters or data are invalid.
# {RE3.1} Validates consistency between `felm` and base R `lm` in terms of coefficients, R-squared, and fitted values.
# {RE3.2} Compares model outputs against established benchmarks such as base R's `lm`.
# {RE5.1} Validates appropriate error handling for omitted arguments or missing data.
# {RE6.0} Implements robust testing for invalid or collinear regressors.
# {RE7.1} Validates that proportional regressors or collinear terms are detected and produce errors.
# {RE7.1a} Adding noise to the depending variable minimally affects the speed. I tested that explicitly.
# {RE7.2} Confirms that model computations remain consistent when small noise is added to data.
# {RE8.1} Ensures computational times remain consistent under similar model specifications.

source(system.file("tinytest", "helper.R", package = "capybara"))

local({
  if (Sys.getenv("CAPYBARA_FULL_TESTING") != "yes") {
    return(NULL)
  }

  # 1FE Stammann centering ----

  ctrl <- list(centering = "stammann")

  ross2004_subset <- ross2004[ross2004$year == 1999, ]
  ross2004_subset <- ross2004_subset[ross2004_subset$ltrade > quantile(ross2004_subset$ltrade, 0.75), ]

  m1 <- felm(formula = ltrade ~ ldist | ctry1, data = ross2004_subset, control = ctrl)
  m2 <- lm(ltrade ~ ldist + as.factor(ctry1), ross2004_subset)

  expect_equal(coef(m1), coef(m2)[2], tolerance = 1e-2)

  n <- nrow(ross2004_subset)
  expect_equal(length(fitted(m1)), n)
  expect_equal(length(predict(m1)), n)
  expect_equal(length(coef(m1)), 1)
  expect_equal(length(coef(summary(m1))), 4)

  m1 <- felm(ltrade ~ ldist + border | ctry1, ross2004_subset, control = ctrl)
  m2 <- lm(ltrade ~ ldist + border + as.factor(ctry1), ross2004_subset)

  expect_equal(coef(m1), coef(m2)[c(2, 3)], tolerance = 1e-2)

  # 2FE Stammann centering ----

  m1 <- felm(ltrade ~ ldist + border | ctry1 + ctry2, ross2004_subset, control = ctrl)
  m2 <- lm(ltrade ~ ldist + border + as.factor(ctry1) + as.factor(ctry2), ross2004_subset)

  expect_equal(coef(m1), coef(m2)[c(2, 3)], tolerance = 1e-2)

  s1 <- summary(m1)
  s2 <- summary(m2)

  expect_equal(s1$r_squared, s2$r.squared, tolerance = 1e-2)
  expect_equal(s1$adj_r_squared, s2$adj.r.squared, tolerance = 1e-2)

  ross2004_subset2 <- ross2004_subset
  ross2004_subset2$ldist[2] <- NA

  m1 <- felm(ltrade ~ ldist + border | ctry1 + ctry2, ross2004_subset2, control = ctrl)
  m2 <- lm(ltrade ~ ldist + border + as.factor(ctry1) + as.factor(ctry2), ross2004_subset2)

  expect_equal(coef(m1), coef(m2)[c(2, 3)], tolerance = 1e-2)

  s1 <- summary(m1)
  s2 <- summary(m2)

  expect_equal(s1$r_squared, s2$r.squared, tolerance = 1e-2)
  expect_equal(s1$adj_r_squared, s2$adj.r.squared, tolerance = 1e-2)

  m1 <- felm(ltrade ~ ldist + border | ctry1 + ctry2 | year, ross2004_subset, control = ctrl)

  expect_equal(coef(m1), coef(m2)[c(2, 3)], tolerance = 1e-2)

  # 3FE Stammann centering ----

  ross2004_subset <- ross2004[ross2004$year %in% c(1994, 1999), ]
  ross2004_subset <- ross2004_subset[ross2004_subset$ltrade > quantile(ross2004_subset$ltrade, 0.75), ]

  m1 <- felm(ltrade ~ ldist + border | ctry1 + ctry2 + year, ross2004_subset, control = ctrl)
  m2 <- lm(
    ltrade ~ ldist + border + as.factor(ctry1) + as.factor(ctry2) + as.factor(year),
    ross2004_subset
  )

  expect_equal(coef(m1), coef(m2)[c(2, 3)], tolerance = 1e-2)

  s1 <- summary(m1)
  s2 <- summary(m2)
  expect_equal(s1$r_squared, s2$r.squared, tolerance = 1e-2)
  expect_equal(s1$adj_r_squared, s2$adj.r.squared, tolerance = 1e-2)

  # proportional regressors return NA coefficients (stammann centering) ----

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

  # felm correctly predicts values outside the inter-quartile range (Stammann centering) ----

  d1 <- ross2004_subset[
    ross2004_subset$ltrade >= quantile(ross2004_subset$ltrade, 0.25) &
      ross2004_subset$ltrade <= quantile(ross2004_subset$ltrade, 0.75),
  ]
  d2 <- ross2004_subset[
    ross2004_subset$ltrade < quantile(ross2004_subset$ltrade, 0.25) |
      ross2004_subset$ltrade > quantile(ross2004_subset$ltrade, 0.75),
  ]

  m1_lm <- felm(ltrade ~ ldist + border | ctry1, ross2004_subset, control = ctrl)
  m2_lm <- lm(ltrade ~ ldist + border + as.factor(ctry1), ross2004_subset)

  pred1_lm <- predict(m1_lm, newdata = d1)
  pred2_lm <- predict(m1_lm, newdata = d2)

  expect_true(mape(d1$ltrade, pred1_lm) < mape(d2$ltrade, pred2_lm))
  expect_equal(pred1_lm, predict(m2_lm, newdata = d1), tolerance = 1e-2)
  expect_equal(pred2_lm, predict(m2_lm, newdata = d2), tolerance = 1e-2)

  # felm with weights works (Stammann centering) ----

  ross2004_subset$trade_pair <- ave(ross2004_subset$ltrade, ross2004_subset$pair,
    FUN = function(x) sum(x, na.rm = TRUE)
  )

  m1 <- felm(ltrade ~ ldist | ctry1, weights = ~trade_pair, data = ross2004_subset, control = ctrl)
  m2 <- felm(ltrade ~ ldist | ctry1, weights = ross2004_subset$trade_pair, data = ross2004_subset, control = ctrl)

  w <- ross2004_subset$trade_pair
  m3 <- felm(ltrade ~ ldist | ctry1, weights = w, data = ross2004_subset, control = ctrl)

  expect_equal(coef(m2), coef(m1))
  expect_equal(coef(m3), coef(m1))

  w <- NULL
  m4 <- felm(ltrade ~ ldist | ctry1, weights = w, data = ross2004_subset, control = ctrl)

  expect_true(coef(m1) != coef(m4))
})
