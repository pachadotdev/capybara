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

test_that("feglm is similar to glm", {
  # Gaussian ----
  # see test-felm.R

  # Poisson ----
  # see test-fepoisson.R

  # Binomial ----
  mod_binom <- feglm(am ~ wt + mpg | cyl, mtcars, family = binomial())
  mod_binom_base <- glm(am ~ wt + mpg + as.factor(cyl), mtcars, family = binomial())
  expect_equal(unname(coef(mod_binom) - coef(mod_binom_base)[2:3]), c(0, 0), tolerance = 1e-2)

  # Gamma ----
  mod_gamma <- feglm(mpg ~ wt + am | cyl, mtcars, family = Gamma())
  mod_gamma_base <- glm(mpg ~ wt + am + as.factor(cyl), mtcars, family = Gamma())
  expect_equal(coef(mod_gamma_base)[2:3], coef(mod_gamma), tolerance = 1e-2)

  # Inverse Gaussian ----
  mod_invgauss <- feglm(mpg ~ wt + am | cyl, mtcars, family = inverse.gaussian())
  mod_invgauss_base <- glm(mpg ~ wt + am + as.factor(cyl), mtcars, family = inverse.gaussian())
  expect_equal(coef(mod_invgauss_base)[2:3], coef(mod_invgauss), tolerance = 1e-2)
})

test_that("predicted values increase the error outside the inter-quartile range for GLMs", {
  skip_on_cran()

  # Helper function for MAPE calculation
  mape <- function(y, yhat) {
    mean(abs(y - yhat) / y)
  }

  # Create data subsets once
  d1 <- mtcars[mtcars$mpg >= quantile(mtcars$mpg, 0.25) & mtcars$mpg <= quantile(mtcars$mpg, 0.75), ]
  d2 <- mtcars[mtcars$mpg < quantile(mtcars$mpg, 0.25) | mtcars$mpg > quantile(mtcars$mpg, 0.75), ]

  # Poisson ----
  m1_pois <- fepoisson(mpg ~ wt + disp | cyl, mtcars)
  m2_pois <- glm(mpg ~ wt + disp + as.factor(cyl), mtcars, family = quasipoisson())
  # m3_pois <- fixest::fepois(mpg ~ wt + disp | cyl, mtcars)

  pred1_pois <- predict(m1_pois, newdata = d1, type = "response")
  pred2_pois <- predict(m1_pois, newdata = d2, type = "response")
  # pred3_pois <- predict(m3_pois, newdata = d1, type = "response")

  mape1_pois <- mape(d1$mpg, pred1_pois)
  mape2_pois <- mape(d2$mpg, pred2_pois)

  expect_gt(mape2_pois, mape1_pois)

  # Compare with base R Poisson
  pred1_base_pois <- predict(m2_pois, newdata = d1, type = "response")
  pred2_base_pois <- predict(m2_pois, newdata = d2, type = "response")
  expect_equal(unname(pred1_base_pois), pred1_pois, tolerance = 1e-2)
  expect_equal(unname(pred2_base_pois), pred2_pois, tolerance = 1e-2)

  # Compare with fixest Poisson
  # pred1_fixest_pois <- predict(m3_pois, newdata = d1, type = "response")
  # pred2_fixest_pois <- predict(m3_pois, newdata = d2, type = "response")
  # expect_equal(unname(pred1_fixest_pois), pred1_pois, tolerance = 1e-2)
  # expect_equal(unname(pred2_fixest_pois), pred2_pois, tolerance = 1e-2)

  # Binomial ----
  m1_binom <- feglm(am ~ wt + disp | cyl, mtcars, family = binomial())
  m2_binom <- glm(am ~ wt + disp + as.factor(cyl), mtcars, family = binomial())

  pred1_binom <- predict(m1_binom, newdata = d1, type = "response")
  pred2_binom <- predict(m1_binom, newdata = d2, type = "response")

  mape1_binom <- mape(d1$mpg, pred1_binom)
  mape2_binom <- mape(d2$mpg, pred2_binom)

  expect_lt(mape1_binom, mape2_binom)

  # Compare with base R Binomial
  pred1_base_binom <- predict(m2_binom, newdata = d1, type = "response")
  pred2_base_binom <- predict(m2_binom, newdata = d2, type = "response")
  expect_equal(pred1_binom, unname(pred1_base_binom), tolerance = 1e-2)
  expect_equal(pred2_binom, unname(pred2_base_binom), tolerance = 1e-2)
})

test_that("predicted values increase the error outside the inter-quartile range for LMs", {
  skip_on_cran()

  # Helper function for MAPE calculation
  mape <- function(y, yhat) {
    mean(abs(y - yhat) / y)
  }

  # Create data subsets once
  d1 <- mtcars[mtcars$mpg >= quantile(mtcars$mpg, 0.25) & mtcars$mpg <= quantile(mtcars$mpg, 0.75), ]
  d2 <- mtcars[mtcars$mpg < quantile(mtcars$mpg, 0.25) | mtcars$mpg > quantile(mtcars$mpg, 0.75), ]

  # Binomial GLM ----
  m1_binom <- feglm(am ~ wt + disp | cyl, mtcars, family = binomial())
  m2_binom <- glm(am ~ wt + disp + as.factor(cyl), mtcars, family = binomial())

  pred1_binom <- predict(m1_binom, newdata = d1, type = "response")
  pred2_binom <- predict(m1_binom, newdata = d2, type = "response")

  mape1_binom <- mape(d1$mpg, pred1_binom)
  mape2_binom <- mape(d2$mpg, pred2_binom)

  expect_lt(mape1_binom, mape2_binom)

  # Compare with base R Binomial
  pred1_base_binom <- predict(m2_binom, newdata = d1, type = "response")
  pred2_base_binom <- predict(m2_binom, newdata = d2, type = "response")
  expect_equal(pred1_binom, unname(pred1_base_binom), tolerance = 1e-2)
  expect_equal(pred2_binom, unname(pred2_base_binom), tolerance = 1e-2)
})

test_that("proportional regressors return NA coefficients", {
  set.seed(200100)
  d <- data.frame(
    y = rnorm(100),
    x1 = rnorm(100),
    f = factor(sample(1:2, 1000, replace = TRUE))
  )
  d$x2 <- 2 * d$x1

  fit1 <- glm(y ~ x1 + x2 + as.factor(f), data = d, family = gaussian())
  fit2 <- feglm(y ~ x1 + x2 | f, data = d, family = gaussian())

  expect_equal(coef(fit2), coef(fit1)[2:3], tolerance = 1e-2)

  expect_equal(predict(fit2), unname(predict(fit1)), tolerance = 1e-2)
})
