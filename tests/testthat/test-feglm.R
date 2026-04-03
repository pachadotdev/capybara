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
  mod_binom_base <- glm(
    am ~ wt + mpg + as.factor(cyl),
    mtcars,
    family = binomial()
  )
  # coef(mod_binom)
  # coef(mod_binom_base)[2:3]

  expect_equal(
    unname(coef(mod_binom) - coef(mod_binom_base)[2:3]),
    c(0, 0),
    tolerance = 1e-2
  )

  mod_binom <- feglm(am ~ wt + mpg | cyl, mtcars, family = binomial())
  # mod_binom_fixest <- fixest::feglm(am ~ wt + mpg | cyl, mtcars, family = binomial())

  mod_binom
  # mod_binom_fixest

  # predict(mod_binom)
  # predict(mod_binom_fixest)

  # Gamma ----
  mod_gamma <- feglm(mpg ~ wt + am | cyl, mtcars, family = Gamma())
  mod_gamma_base <- glm(
    mpg ~ wt + am + as.factor(cyl),
    mtcars,
    family = Gamma()
  )
  expect_equal(coef(mod_gamma_base)[2:3], coef(mod_gamma), tolerance = 1e-2)

  # Inverse Gaussian ----
  mod_invgauss <- feglm(
    mpg ~ wt + am | cyl,
    mtcars,
    family = inverse.gaussian()
  )
  mod_invgauss_base <- glm(
    mpg ~ wt + am + as.factor(cyl),
    mtcars,
    family = inverse.gaussian()
  )
  expect_equal(
    coef(mod_invgauss_base)[2:3],
    coef(mod_invgauss),
    tolerance = 1e-2
  )
})

test_that("feglm works without fixed effects", {
  m1 <- feglm(log(mpg) ~ log(wt), data = mtcars)
  m2 <- glm(log(mpg) ~ log(wt), data = mtcars)

  expect_equal(coef(m1), coef(m2), tolerance = 1e-6)
})

test_that("predicted values increase the error outside the inter-quartile range for GLMs", {
  skip_on_cran()

  # Helper function for MAPE calculation
  mape <- function(y, yhat) {
    mean(abs(y - yhat) / y)
  }

  # Create data subsets once
  d1 <- mtcars[
    mtcars$mpg >= quantile(mtcars$mpg, 0.25) &
      mtcars$mpg <= quantile(mtcars$mpg, 0.75),
  ]
  d2 <- mtcars[
    mtcars$mpg < quantile(mtcars$mpg, 0.25) |
      mtcars$mpg > quantile(mtcars$mpg, 0.75),
  ]

  # Poisson ----
  m1_pois <- fepoisson(mpg ~ wt + disp | cyl, mtcars, control = fit_control(return_fe = TRUE))
  m2_pois <- glm(
    mpg ~ wt + disp + as.factor(cyl),
    mtcars,
    family = quasipoisson()
  )
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
  expect_equal(pred1_base_pois, pred1_pois, tolerance = 1e-2)
  expect_equal(pred2_base_pois, pred2_pois, tolerance = 1e-2)

  # Compare with fixest Poisson
  # pred1_fixest_pois <- predict(m3_pois, newdata = d1, type = "response")
  # pred2_fixest_pois <- predict(m3_pois, newdata = d2, type = "response")
  # expect_equal(unname(pred1_fixest_pois), pred1_pois, tolerance = 1e-2)
  # expect_equal(unname(pred2_fixest_pois), pred2_pois, tolerance = 1e-2)

  # Binomial ----
  m1_binom <- feglm(am ~ wt + disp | cyl, mtcars, family = binomial(), control = fit_control(return_fe = TRUE))
  m2_binom <- glm(am ~ wt + disp + as.factor(cyl), mtcars, family = binomial())

  pred1_binom <- predict(m1_binom, newdata = d1, type = "response")
  pred2_binom <- predict(m1_binom, newdata = d2, type = "response")

  mape1_binom <- mape(d1$mpg, pred1_binom)
  mape2_binom <- mape(d2$mpg, pred2_binom)

  expect_lt(mape1_binom, mape2_binom)

  # Compare with base R Binomial
  pred1_base_binom <- predict(m2_binom, newdata = d1, type = "response")
  pred2_base_binom <- predict(m2_binom, newdata = d2, type = "response")
  expect_equal(pred1_binom, pred1_base_binom, tolerance = 1e-2)
  expect_equal(pred2_binom, pred2_base_binom, tolerance = 1e-2)

  names(m2_binom)
})

test_that("predicted values increase the error outside the inter-quartile range for LMs", {
  skip_on_cran()

  # Helper function for MAPE calculation
  mape <- function(y, yhat) {
    mean(abs(y - yhat) / y)
  }

  # Create data subsets once
  d1 <- mtcars[
    mtcars$mpg >= quantile(mtcars$mpg, 0.25) &
      mtcars$mpg <= quantile(mtcars$mpg, 0.75),
  ]
  d2 <- mtcars[
    mtcars$mpg < quantile(mtcars$mpg, 0.25) |
      mtcars$mpg > quantile(mtcars$mpg, 0.75),
  ]

  # Binomial GLM ----

  m1_binom <- feglm(am ~ wt + disp | cyl, mtcars, family = binomial(), control = fit_control(return_fe = TRUE))
  # m2_binom <- fixest::feglm(am ~ wt + disp | cyl, mtcars, family = binomial())
  m2_binom <- glm(am ~ wt + disp + as.factor(cyl), mtcars, family = binomial())

  # coef(m1_binom)
  # coef(m2_binom)[2:3]

  pred1_binom <- predict(m1_binom, newdata = d1, type = "response")
  pred2_binom <- predict(m1_binom, newdata = d2, type = "response")

  mape1_binom <- mape(d1$mpg, pred1_binom)
  mape2_binom <- mape(d2$mpg, pred2_binom)

  expect_lt(mape1_binom, mape2_binom)

  # Compare with base R Binomial
  pred1_base_binom <- predict(m2_binom, newdata = d1, type = "response")
  pred2_base_binom <- predict(m2_binom, newdata = d2, type = "response")
  expect_equal(pred1_binom, pred1_base_binom, tolerance = 1e-2)
  expect_equal(pred2_binom, pred2_base_binom, tolerance = 1e-2)
})

test_that("proportional regressors return NA coefficients", {
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

test_that("feglm with weights works", {
  skip_on_cran()

  m1 <- feglm(mpg ~ wt | am, weights = ~cyl, data = mtcars)
  m2 <- feglm(mpg ~ wt | am, weights = mtcars$cyl, data = mtcars)

  w <- mtcars$cyl
  m3 <- feglm(mpg ~ wt | am, weights = w, data = mtcars)

  expect_equal(coef(m2), coef(m1))
  expect_equal(coef(m3), coef(m1))

  w <- NULL
  m4 <- feglm(mpg ~ wt | am, weights = w, data = mtcars)

  expect_gt(coef(m1), coef(m4))
})

# Stammann centering ----

test_that("feglm is similar to glm (stammann centering)", {
  ctrl <- list(centering = "stammann")

  # Binomial ----
  mod_binom <- feglm(am ~ wt + mpg | cyl, mtcars, family = binomial(), control = ctrl)
  mod_binom_base <- glm(
    am ~ wt + mpg + as.factor(cyl),
    mtcars,
    family = binomial()
  )

  expect_equal(
    unname(coef(mod_binom) - coef(mod_binom_base)[2:3]),
    c(0, 0),
    tolerance = 1e-2
  )

  # Gamma ----
  mod_gamma <- feglm(mpg ~ wt + am | cyl, mtcars, family = Gamma(), control = ctrl)
  mod_gamma_base <- glm(
    mpg ~ wt + am + as.factor(cyl),
    mtcars,
    family = Gamma()
  )
  expect_equal(coef(mod_gamma_base)[2:3], coef(mod_gamma), tolerance = 1e-2)

  # Inverse Gaussian ----
  mod_invgauss <- feglm(
    mpg ~ wt + am | cyl,
    mtcars,
    family = inverse.gaussian(),
    control = ctrl
  )
  mod_invgauss_base <- glm(
    mpg ~ wt + am + as.factor(cyl),
    mtcars,
    family = inverse.gaussian()
  )
  expect_equal(
    coef(mod_invgauss_base)[2:3],
    coef(mod_invgauss),
    tolerance = 1e-2
  )
})

test_that("feglm works without fixed effects (stammann centering)", {
  # centering is unused without FEs, but control must be accepted
  ctrl <- list(centering = "stammann")
  m1 <- feglm(log(mpg) ~ log(wt), data = mtcars, control = ctrl)
  m2 <- glm(log(mpg) ~ log(wt), data = mtcars)

  expect_equal(coef(m1), coef(m2), tolerance = 1e-6)
})

test_that("proportional regressors return NA coefficients (stammann centering)", {
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

test_that("feglm with weights works (stammann centering)", {
  skip_on_cran()
  ctrl <- list(centering = "stammann")

  m1 <- feglm(mpg ~ wt | am, weights = ~cyl, data = mtcars, control = ctrl)
  m2 <- feglm(mpg ~ wt | am, weights = mtcars$cyl, data = mtcars, control = ctrl)

  w <- mtcars$cyl
  m3 <- feglm(mpg ~ wt | am, weights = w, data = mtcars, control = ctrl)

  expect_equal(coef(m2), coef(m1))
  expect_equal(coef(m3), coef(m1))

  w <- NULL
  m4 <- feglm(mpg ~ wt | am, weights = w, data = mtcars, control = ctrl)

  expect_gt(coef(m1), coef(m4))
})

# Average Partial Effects (APE) tests ----

test_that("feglm computes APEs for binomial models", {
  # Fit binomial model with APE computation
  mod_binom_ape <- feglm(
    am ~ wt + mpg | cyl,
    mtcars,
    family = binomial(),
    control = fit_control(compute_apes = TRUE)
  )

  # Check that APE results are present
  expect_true(!is.null(mod_binom_ape$ape_delta))
  expect_true(!is.null(mod_binom_ape$ape_vcov))
  expect_true(!is.null(mod_binom_ape$ape_binary))

  # Check dimensions
  expect_equal(length(mod_binom_ape$ape_delta), 2) # wt and mpg
  expect_equal(dim(mod_binom_ape$ape_vcov), c(2, 2))
  expect_equal(length(mod_binom_ape$ape_binary), 2)

  # APEs should be finite

  expect_true(all(is.finite(mod_binom_ape$ape_delta)))

  # APEs should have reasonable magnitudes (between -1 and 1 for probabilities)
  expect_true(all(abs(mod_binom_ape$ape_delta) < 1))

  # Variance matrix should be symmetric and positive semi-definite
  expect_equal(mod_binom_ape$ape_vcov, t(mod_binom_ape$ape_vcov), tolerance = 1e-10)
  expect_true(all(eigen(mod_binom_ape$ape_vcov)$values >= -1e-10))
})

test_that("APEs are not computed when compute_apes = FALSE", {
  # Default: compute_apes = FALSE
  mod_binom <- feglm(
    am ~ wt + mpg | cyl,
    mtcars,
    family = binomial()
  )

  # APE results should not be present
  expect_null(mod_binom$ape_delta)
  expect_null(mod_binom$ape_vcov)
})

test_that("APEs are not computed for non-binomial models", {
  # Poisson model with compute_apes = TRUE should not have APEs
  mod_poisson <- feglm(
    mpg ~ wt + am | cyl,
    mtcars,
    family = poisson(),
    control = fit_control(compute_apes = TRUE)
  )

  # APE results should not be present for non-binomial
  expect_null(mod_poisson$ape_delta)
})

test_that("APE binary indicator correctly identifies binary regressors", {
  # Create data with mixed binary and continuous regressors
  set.seed(123)
  d <- data.frame(
    y = rbinom(100, 1, 0.5),
    x_cont = rnorm(100),
    x_binary = sample(0:1, 100, replace = TRUE),
    f = factor(sample(1:3, 100, replace = TRUE))
  )

  mod <- feglm(
    y ~ x_cont + x_binary | f,
    d,
    family = binomial(),
    control = fit_control(compute_apes = TRUE)
  )

  # Check binary indicator
  expect_equal(length(mod$ape_binary), 2)
  # x_cont should be 0 (continuous), x_binary should be 1 (binary)
  expect_equal(as.numeric(mod$ape_binary), c(0, 1))
})

test_that("APE standard errors can be computed from vcov", {
  mod <- feglm(
    am ~ wt + mpg | cyl,
    mtcars,
    family = binomial(),
    control = fit_control(compute_apes = TRUE)
  )

  # Compute standard errors from variance-covariance matrix
  ape_se <- sqrt(diag(mod$ape_vcov))

  # Standard errors should be finite and positive
  expect_true(all(is.finite(ape_se)))
  expect_true(all(ape_se > 0))

  # z-statistics
  z_vals <- mod$ape_delta / ape_se
  expect_true(all(is.finite(z_vals)))
})

test_that("APE with finite population correction works", {
  # With n_pop > n, variance should be adjusted
  mod_no_pop <- feglm(
    am ~ wt + mpg | cyl,
    mtcars,
    family = binomial(),
    control = fit_control(compute_apes = TRUE)
  )

  mod_with_pop <- feglm(
    am ~ wt + mpg | cyl,
    mtcars,
    family = binomial(),
    control = fit_control(compute_apes = TRUE, ape_n_pop = 1000)
  )

  # Both should have APE results
  expect_true(!is.null(mod_no_pop$ape_delta))
  expect_true(!is.null(mod_with_pop$ape_delta))

  # APE point estimates should be similar (n_pop doesn't affect point estimates)
  expect_equal(mod_no_pop$ape_delta, mod_with_pop$ape_delta, tolerance = 1e-10)

  # Variance matrices may differ due to finite population correction
  # (the correction adds additional variance terms)
  expect_true(!is.null(mod_with_pop$ape_vcov))
})

test_that("APE panel_structure parameter works", {
  # Create a simple two-way panel dataset
  set.seed(42)
  n_obs <- 200
  d <- data.frame(
    y = rbinom(n_obs, 1, 0.5),
    x = rnorm(n_obs),
    fe1 = factor(rep(1:10, each = 20)),
    fe2 = factor(rep(1:20, times = 10))
  )

  # Classic panel structure
  mod_classic <- feglm(
    y ~ x | fe1 + fe2,
    d,
    family = binomial(),
    control = fit_control(
      compute_apes = TRUE,
      ape_panel_structure = "classic"
    )
  )

  # Network panel structure
  mod_network <- feglm(
    y ~ x | fe1 + fe2,
    d,
    family = binomial(),
    control = fit_control(
      compute_apes = TRUE,
      ape_panel_structure = "network"
    )
  )

  # Both should produce valid APE results
  expect_true(!is.null(mod_classic$ape_delta))
  expect_true(!is.null(mod_network$ape_delta))
  expect_true(all(is.finite(mod_classic$ape_delta)))
  expect_true(all(is.finite(mod_network$ape_delta)))
})

test_that("APE with sampling_fe parameter works", {
  set.seed(123)
  d <- data.frame(
    y = rbinom(100, 1, 0.5),
    x = rnorm(100),
    f = factor(sample(1:5, 100, replace = TRUE))
  )

  # Independence sampling assumption
  mod_indep <- feglm(
    y ~ x | f,
    d,
    family = binomial(),
    control = fit_control(
      compute_apes = TRUE,
      ape_n_pop = 500,
      ape_sampling_fe = "independence"
    )
  )

  # Unrestricted sampling assumption
  mod_unrest <- feglm(
    y ~ x | f,
    d,
    family = binomial(),
    control = fit_control(
      compute_apes = TRUE,
      ape_n_pop = 500,
      ape_sampling_fe = "unrestricted"
    )
  )

  # Both should produce valid APE results
  expect_true(!is.null(mod_indep$ape_delta))
  expect_true(!is.null(mod_unrest$ape_delta))

  # Point estimates should be the same
  expect_equal(mod_indep$ape_delta, mod_unrest$ape_delta, tolerance = 1e-10)
})

test_that("APE with weak_exo parameter works", {
  set.seed(456)
  d <- data.frame(
    y = rbinom(100, 1, 0.5),
    x = rnorm(100),
    f = factor(sample(1:5, 100, replace = TRUE))
  )

  # Strict exogeneity (default)
  mod_strict <- feglm(
    y ~ x | f,
    d,
    family = binomial(),
    control = fit_control(
      compute_apes = TRUE,
      ape_n_pop = 500,
      ape_weak_exo = FALSE
    )
  )

  # Weak exogeneity
  mod_weak <- feglm(
    y ~ x | f,
    d,
    family = binomial(),
    control = fit_control(
      compute_apes = TRUE,
      ape_n_pop = 500,
      ape_weak_exo = TRUE
    )
  )

  # Both should produce valid APE results
  expect_true(!is.null(mod_strict$ape_delta))
  expect_true(!is.null(mod_weak$ape_delta))

  # Point estimates should be the same (weak_exo only affects variance)
  expect_equal(mod_strict$ape_delta, mod_weak$ape_delta, tolerance = 1e-10)
})
