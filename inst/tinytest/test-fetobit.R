#' srr_stats (tests)
#' @srrstats {RE3.1} Validates consistency between `fetobit` and other established R models.
#' @srrstats {RE3.2} Compares coefficients produced by `fetobit` with baseline models.
#' @srrstats {RE4.3} Ensures stable estimates when adding negligible noise to the data.
#' @srrstats {RE5.1} Validates proper output generation for the model summary and printing methods.
#' @srrstats {RE7.3} Confirms that estimated coefficients are within a reasonable variation threshold.
#' @noRd
NULL

source(system.file("tinytest", "helper.R", package = "capybara"))

# Helper to create tobit test data
make_tobit_data <- function(n = 200, lower = 0, upper = Inf, seed = 123) {
  set.seed(seed)
  d <- data.frame(
    x1 = rnorm(n),
    x2 = rnorm(n),
    f1 = factor(sample(1:3, n, replace = TRUE)),
    f2 = factor(sample(1:2, n, replace = TRUE))
  )
  # Generate latent y*
  y_star <- 2 + 1.5 * d$x1 - 0.8 * d$x2 + rnorm(n)
  # Apply censoring
  d$y <- pmin(pmax(y_star, lower), upper)
  d
}

# fetobit basic functionality works"
local({
  d <- make_tobit_data(n = 200, lower = 0, seed = 42)

  # Should run without error
  mod <- fetobit(y ~ x1 | f1, d, tobit_lb = 0)

  expect_true(inherits(mod, "feglm"))
  expect_equal(length(coef(mod)), 1)
  
  # Check that fitted values are returned
  n_obs <- unname(mod[["nobs"]]["nobs_full"])
  expect_equal(length(fitted(mod)), n_obs)
  expect_equal(length(predict(mod)), n_obs)
})

# fetobit without FE is similar to AER::tobit"
local({
  skip_on_cran()

  # Create data without FE for comparison (AER::tobit doesn't support FE)
  set.seed(123)
  d <- data.frame(
    x1 = rnorm(300),
    x2 = rnorm(300)
  )
  y_star <- 2 + 1.5 * d$x1 - 0.8 * d$x2 + rnorm(300)
  d$y <- pmax(y_star, 0)

  # Fit with AER
  # mod_aer <- AER::tobit(y ~ 1 + x1 + x2, data = d, left = 0)
  # coef(mod_aer)
  coef_mod_aer <- c(2.014803, 1.455198, -0.852519)

  # Fit with fetobit (no FE)
  mod_cap <- fetobit(y ~ x1 + x2, d, tobit_lb = 0)

  expect_equal(coef_mod_aer, unname(coef(mod_cap)), tolerance = 0.05)
})

# fetobit handles two-sided censoring"
local({
  d <- make_tobit_data(n = 200, lower = 0, upper = 5, seed = 456)

  mod <- fetobit(y ~ x1 | f1, d, tobit_lb = 0, tobit_ub = 5)

  expect_true(inherits(mod, "feglm"))
  expect_equal(length(coef(mod)), 1)

  # Fitted values should be within valid range (though for tobit this isn't strictly required)
  expect_true(all(is.finite(fitted(mod))))
})

# fetobit with multiple fixed effects"
local({
  d <- make_tobit_data(n = 300, lower = 0, seed = 222)

  # K = 2
  mod <- fetobit(y ~ x1 | f1 + f2, d, tobit_lb = 0)

  expect_true(inherits(mod, "feglm"))
  expect_equal(length(coef(mod)), 1)
})

# fetobit handles cluster standard errors"
local({
  d <- make_tobit_data(n = 200, lower = 0, seed = 111)
  d$cl <- factor(sample(1:10, nrow(d), replace = TRUE))

  mod <- fetobit(y ~ x1 | f1 | cl, d, tobit_lb = 0, vcov = "cluster")
  smod <- summary(mod)

  expect_equal(mod$vcov_type, "cluster")
  expect_true(all(is.finite(coef(smod)[, "Std. Error"])))
})

# fetobit estimation is stable with noise"
local({
  set.seed(123)
  d <- make_tobit_data(n = 200, lower = 0, seed = 456)
  d$x1_noisy <- d$x1 + pmax(rnorm(nrow(d)), 0) * .Machine$double.eps

  m1 <- fetobit(y ~ x1 | f1, d, tobit_lb = 0)
  m2 <- fetobit(y ~ x1_noisy | f1, d, tobit_lb = 0)

  expect_equal(unname(coef(m1)), unname(coef(m2)))
})

# fetobit with FE is similar to AER::tobit with dummies"
local({
  skip_on_cran()

  d <- make_tobit_data(n = 300, lower = 0, seed = 555)
  d$f1 <- factor(sample(1:5, nrow(d), replace = TRUE))

  # Fit with AER (include FE as dummies)
  # mod_aer <- AER::tobit(y ~ 0 + x1 + as.factor(f1), data = d, left = 0)
  # coef(mod_aer)
  coef_mod_aer <- c(1.525423, 1.950605, 2.074851, 1.955391, 2.053087, 1.958672)

  # Fit with fetobit
  mod_cap <- fetobit(y ~ x1 | f1, d, tobit_lb = 0, control = fit_control(return_fe = TRUE))

  # Compare x1 coefficient
  expect_true(abs(coef_mod_aer[1] - coef(mod_cap)["x1"]) < 0.05)
})
