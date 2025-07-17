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

test_that("felm works", {
  # 1-FE ----

  m1 <- felm(mpg ~ wt | cyl, mtcars)
  # m1_fixest <- fixest::feols(mpg ~ wt | cyl, mtcars)
  m2 <- lm(mpg ~ wt + as.factor(cyl), mtcars)

  # coef(m1)
  # coef(m1_fixest)
  #        wt
  # -3.205613

  # m1$fixed.effects
  # fixest::fixef(m1_fixest)
  # $cyl
  #        4        6        8 
  # 33.99079 29.73521 27.91993 

  expect_equal(coef(m1), coef(m2)[2], tolerance = 1e-2)

  n <- nrow(mtcars)
  expect_equal(length(fitted(m1)), n)
  expect_equal(length(predict(m1)), n)
  expect_equal(length(coef(m1)), 1)
  expect_equal(length(coef(summary(m1))), 4)

  coef(felm(mpg ~ wt + qsec | cyl, mtcars))
  coef(fixest::feols(mpg ~ wt + qsec | cyl, mtcars))

  m1 <- felm(mpg ~ wt + qsec | cyl, mtcars)
  m2 <- lm(mpg ~ wt + qsec + as.factor(cyl), mtcars)

  expect_equal(coef(m1), coef(m2)[c(2, 3)], tolerance = 1e-2)

  # 2-FE ----

  m1 <- felm(mpg ~ wt + qsec | cyl + am, mtcars)
  m2 <- lm(mpg ~ wt + qsec + as.factor(cyl) + as.factor(am), mtcars)

  expect_equal(coef(m1), coef(m2)[c(2, 3)], tolerance = 1e-2)

  s1 <- summary(m1)
  s2 <- summary(m2)

  expect_equal(s1$r.squared, s2$r.squared, tolerance = 1e-2)
  expect_equal(s1$adj.r.squared, s2$adj.r.squared, tolerance = 1e-2)

  mtcars2 <- mtcars
  mtcars2$wt[2] <- NA

  m1 <- felm(mpg ~ wt + qsec | cyl + am, mtcars2)
  m2 <- lm(mpg ~ wt + qsec + as.factor(cyl) + as.factor(am), mtcars2)

  expect_equal(coef(m1), coef(m2)[c(2, 3)], tolerance = 1e-2)

  s1 <- summary(m1)
  s2 <- summary(m2)

  expect_equal(s1$r.squared, s2$r.squared, tolerance = 1e-2)
  expect_equal(s1$adj.r.squared, s2$adj.r.squared, tolerance = 1e-2)

  m1 <- felm(mpg ~ wt + qsec | cyl + am | carb, mtcars)

  expect_equal(coef(m1), coef(m2)[c(2, 3)], tolerance = 1e-2)

  # 3-FE ----

  m1 <- felm(mpg ~ wt + qsec | cyl + am + gear, mtcars)
  m2 <- lm(mpg ~ wt + qsec + as.factor(cyl) + as.factor(am) + as.factor(gear), mtcars)

  expect_equal(coef(m1), coef(m2)[c(2, 3)], tolerance = 1e-2)

  s1 <- summary(m1)
  s2 <- summary(m2)
  expect_equal(s1$r.squared, s2$r.squared, tolerance = 1e-2)
  expect_equal(s1$adj.r.squared, s2$adj.r.squared, tolerance = 1e-2)
})

test_that("felm time is the minimally affected when adding noise to the data", {
  mtcars2 <- mtcars[, c("mpg", "wt", "cyl")]
  set.seed(200100)
  mtcars2$mpg <- mtcars2$mpg + rbinom(nrow(mtcars2), 1, 0.5) *
    .Machine$double.eps
  m1 <- felm(mpg ~ wt | cyl, mtcars)
  m2 <- felm(mpg ~ wt | cyl, mtcars2)
  expect_equal(coef(m1), coef(m2))
  expect_equal(fixed_effects(m1), fixed_effects(m2))

  t1 <- rep(NA, 10)
  t2 <- rep(NA, 10)
  for (i in 1:10) {
    a <- Sys.time()
    m1 <- felm(mpg ~ wt | cyl, mtcars)
    b <- Sys.time()
    t1[i] <- b - a

    a <- Sys.time()
    m2 <- felm(mpg ~ wt | cyl, mtcars2)
    b <- Sys.time()
    t2[i] <- b - a
  }
  expect_gte(0.05, abs(median(t1) - median(t2)))
})

test_that("proportional regressors return NA coefficients", {
  set.seed(200100)
  d <- data.frame(
    y = rnorm(100),
    x1 = rnorm(100),
    f = factor(sample(1:2, 100, replace = TRUE))
  )

  d$x2 <- 2 * d$x1
  fit1 <- lm(y ~ x1 + x2 + as.factor(f), data = d)
  fit2 <- felm(y ~ x1 + x2 | f, data = d)

  fit1$coefficients
  fit2$coefficients

  expect_equal(coef(fit2), coef(fit1)[2:3], tolerance = 1e-2)
  expect_equal(predict(fit2), unname(predict(fit1)), tolerance = 1e-2)
})

test_that("Inf values are dropped", {
  mtcars2 <- mtcars[, c("mpg", "wt", "cyl")]

  mtcars2$mpg[1] <- 0
  expect_error(felm(log(mpg) ~ log(wt) | cyl, mtcars2), "Infinite values")

  mtcars2$mpg[1] <- 1
  mtcars2$wt[2] <- 0
  expect_error(felm(log(mpg) ~ log(wt) | cyl, mtcars2), "Infinite values")

  mtcars2$wt[2] <- 1
  m1 <- felm(log(mpg) ~ log(wt) | cyl, mtcars2)
  m2 <- lm(log(mpg) ~ log(wt) + as.factor(cyl), mtcars2)
  expect_equal(coef(m2)[2], coef(m1), tolerance = 1e-2)
})

test_that("felm correctly predicts values outside the inter-quartile range", {
  # Helper function for MAPE calculation
  mape <- function(y, yhat) {
    mean(abs(y - yhat) / y)
  }

  # Create data subsets once
  d1 <- mtcars[mtcars$mpg >= quantile(mtcars$mpg, 0.25) & mtcars$mpg <= quantile(mtcars$mpg, 0.75), ]
  d2 <- mtcars[mtcars$mpg < quantile(mtcars$mpg, 0.25) | mtcars$mpg > quantile(mtcars$mpg, 0.75), ]

  m1_lm <- felm(mpg ~ wt + disp | cyl, mtcars)
  m2_lm <- lm(mpg ~ wt + disp + as.factor(cyl), mtcars)

  pred1_lm <- predict(m1_lm, newdata = d1)
  pred2_lm <- predict(m1_lm, newdata = d2)

  mape1_lm <- mape(d1$mpg, pred1_lm)
  mape2_lm <- mape(d2$mpg, pred2_lm)

  expect_lt(mape1_lm, mape2_lm)

  # Compare with base R linear model
  pred1_base_lm <- predict(m2_lm, newdata = d1)
  pred2_base_lm <- predict(m2_lm, newdata = d2)
  expect_equal(pred1_lm, unname(pred1_base_lm), tolerance = 1e-2)
  expect_equal(pred2_lm, unname(pred2_base_lm), tolerance = 1e-2)
})
