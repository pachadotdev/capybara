#' srr_stats (tests)
#' @srrstats {RE3.1} Validates consistency between `feprobit` and other established R models like `glm` with comparable families.
#' @srrstats {RE3.2} Compares coefficients produced by `feprobit` with those from base R models to validate similarity.
#' @srrstats {RE4.3} Ensures stable estimates when adding negligible noise to the data.
#' @srrstats {RE5.1} Validates proper output generation for the model summary and printing methods.
#' @srrstats {RE7.3} Confirms that estimated coefficients are within a reasonable variation threshold compared to baseline models.
#' @srrstats {G5.10} The CAPYBARA_EXTENDED_TESTS environment variable can be set to true to run extended tests.
#' @srrstats {G5.11} The extended tests do not require additional downloads.
#' @srrstats {G5.11a} As for G5.11., the extended tests do not require additional downloads.
#' @srrstats {G5.12} The extended tests verify that the algorithm fitting time is robust to noise. This has to be tested with a larger dataset to see that time(clean) <= time(noisy).
#' @noRd
NULL

# Helper to create probit test data without separation issues
make_probit_data <- function(n = 200, seed = 123) {
  set.seed(seed)
  d <- data.frame(
    x1 = rnorm(n),
    x2 = rnorm(n),
    f1 = factor(sample(1:3, n, replace = TRUE)),
    f2 = factor(sample(1:2, n, replace = TRUE)),
    f3 = factor(sample(1:2, n, replace = TRUE))
  )
  # Generate y with moderate probability (avoid separation)
  eta <- 0.5 + 0.8 * d$x1 - 0.5 * d$x2
  d$y <- rbinom(n, 1, pnorm(eta))
  d
}

test_that("feprobit is similar to base", {
  skip_on_cran()

  d <- make_probit_data(n = 300, seed = 42)


  # K = 1

  mod <- feprobit(y ~ x1 | f1, d, control = fit_control(return_fe = TRUE))

  mod_base <- glm(
    y ~ x1 + as.factor(f1),
    d,
    family = binomial(link = "probit")
  )

  coef_dist_base <- coef(mod_base)[2]

  dist_variation <- unname(abs((coef(mod)[1] - coef_dist_base) / coef(mod)[1]))

  expect_equal(dist_variation, 0.0, tolerance = 1e-2)

  expect_output(print(mod))

  expect_visible(summary(mod))

  n_obs <- unname(mod[["nobs"]]["nobs_full"])
  expect_equal(length(fitted(mod)), n_obs)
  expect_equal(length(predict(mod)), n_obs)
  expect_equal(length(coef(mod)), 1)

  smod <- summary(mod)

  expect_equal(length(coef(smod)[, 1]), 1)
  expect_output(summary_formula_(smod))
  expect_output(summary_family_(smod))
  expect_output(summary_estimates_(smod, 3))
  expect_output(summary_r2_(smod, 3))
  expect_output(summary_nobs_(smod))
  expect_output(summary_fisher_(smod))

  # K = 2

  mod <- feprobit(y ~ x1 | f1 + f2, d, control = fit_control(return_fe = TRUE))

  mod_base <- glm(
    y ~ x1 + as.factor(f1) + as.factor(f2),
    d,
    family = binomial(link = "probit")
  )

  coef_dist_base <- coef(mod_base)[2]

  dist_variation <- abs((coef(mod)[1] - coef_dist_base) / coef(mod)[1])

  expect_lt(dist_variation, 0.05)

  # K = 3

  mod <- feprobit(y ~ x1 | f1 + f2 + f3, d, control = fit_control(return_fe = TRUE))

  mod_base <- glm(
    y ~ x1 + as.factor(f1) + as.factor(f2) + as.factor(f3),
    d,
    family = binomial(link = "probit")
  )

  coef_dist_base <- coef(mod_base)[2]

  dist_variation <- abs((coef(mod)[1] - coef_dist_base) / coef(mod)[1])

  expect_lt(dist_variation, 0.05)

  expect_equal(mod[["fitted_values"]], mod_base[["fitted.values"]], tolerance = 1e-2)

  pred_mod <- predict(mod, type = "response")
  pred_mod_base <- predict(mod_base, type = "response")

  pred_mod_link <- predict(mod, type = "link")
  pred_mod_base_link <- predict(mod_base, type = "link")

  expect_equal(pred_mod, pred_mod_base, tolerance = 1e-2)
  expect_equal(pred_mod_link, pred_mod_base_link, tolerance = 1e-2)

  pred_mod <- predict(mod, type = "response", newdata = d[1:10, ])
  pred_mod_base <- predict(mod_base, type = "response", newdata = d[1:10, ])

  pred_mod_link <- predict(mod, type = "link", newdata = d[1:10, ])
  pred_mod_base_link <- predict(mod_base, type = "link", newdata = d[1:10, ])

  expect_equal(unname(pred_mod), unname(pred_mod_base), tolerance = 1e-2)
  expect_equal(unname(pred_mod_link), unname(pred_mod_base_link), tolerance = 1e-2)
})

test_that("feprobit estimation is the same adding noise to the data", {
  set.seed(123)

  d <- make_probit_data(n = 200, seed = 456)
  d$x1_noisy <- d$x1 + pmax(rnorm(nrow(d)), 0) * .Machine$double.eps

  m1 <- feprobit(y ~ x1 | f1, d)
  m2 <- feprobit(y ~ x1_noisy | f1, d)

  expect_equal(unname(coef(m1)), unname(coef(m2)))
  expect_equal(m1$fixed.effects, m2$fixed.effects)
})

test_that("proportional regressors return NA coefficients", {
  set.seed(200100)
  d <- data.frame(
    y = rbinom(100, 1, 0.5),
    x1 = rnorm(100),
    f = factor(sample(1:2, 100, replace = TRUE))
  )
  d$x2 <- 2 * d$x1

  fit1 <- glm(y ~ x1 + x2 + as.factor(f), data = d, family = binomial(link = "probit"))
  fit2 <- feprobit(y ~ x1 + x2 | f, data = d)

  expect_equal(coef(fit2), coef(fit1)[2:3], tolerance = 1e-2)
  expect_equal(predict(fit2), predict(fit1, type = "response"), tolerance = 1e-2)
})

test_that("feprobit and feglm with probit() give the same results", {
  d <- make_probit_data(n = 200, seed = 789)

  mod1 <- feprobit(y ~ x1 | f1, d)
  mod2 <- feglm(y ~ x1 | f1, d, family = "probit")
  mod3 <- feglm(y ~ x1 | f1, d, family = binomial(link = "probit"))

  expect_equal(coef(mod1), coef(mod2))
  expect_equal(coef(mod1), coef(mod3))
  expect_equal(fitted(mod1), fitted(mod2))
  expect_equal(fitted(mod1), fitted(mod3))
})

test_that("feprobit handles cluster standard errors", {
  d <- make_probit_data(n = 200, seed = 111)
  d$cl <- factor(sample(1:10, nrow(d), replace = TRUE))

  mod <- feprobit(y ~ x1 | f1 | cl, d, vcov = "cluster")
  smod <- summary(mod)

  expect_equal(mod$vcov_type, "cluster")
  expect_true(all(is.finite(coef(smod)[, "Std. Error"])))
})

# Stammann centering ----

test_that("feprobit is similar to base (stammann centering)", {
  skip_on_cran()
  ctrl <- fit_control(centering = "stammann", return_fe = TRUE)

  d <- make_probit_data(n = 300, seed = 222)

  # K = 1

  mod <- feprobit(y ~ x1 | f1, d, control = ctrl)

  mod_base <- glm(
    y ~ x1 + as.factor(f1),
    d,
    family = binomial(link = "probit")
  )

  dist_variation <- unname(abs(
    (coef(mod)[1] - coef(mod_base)[2]) / coef(mod)[1]
  ))

  expect_equal(dist_variation, 0.0, tolerance = 1e-2)

  n_obs <- unname(mod[["nobs"]]["nobs_full"])
  expect_equal(length(fitted(mod)), n_obs)
  expect_equal(length(predict(mod)), n_obs)
  expect_equal(length(coef(mod)), 1)

  # K = 2

  mod <- feprobit(y ~ x1 | f1 + f2, d, control = ctrl)

  mod_base <- glm(
    y ~ x1 + as.factor(f1) + as.factor(f2),
    d,
    family = binomial(link = "probit")
  )

  dist_variation <- abs((coef(mod)[1] - coef(mod_base)[2]) / coef(mod)[1])

  expect_lt(dist_variation, 0.05)

  # K = 3

  mod <- feprobit(y ~ x1 | f1 + f2 + f3, d, control = ctrl)

  mod_base <- glm(
    y ~ x1 + as.factor(f1) + as.factor(f2) + as.factor(f3),
    d,
    family = binomial(link = "probit")
  )

  dist_variation <- abs((coef(mod)[1] - coef(mod_base)[2]) / coef(mod)[1])

  expect_lt(dist_variation, 0.05)

  expect_equal(mod[["fitted_values"]], mod_base[["fitted.values"]], tolerance = 1e-2)

  pred_mod <- predict(mod, type = "response")
  pred_mod_base <- predict(mod_base, type = "response")
  expect_equal(pred_mod, pred_mod_base, tolerance = 1e-2)

  pred_mod <- predict(mod, type = "response", newdata = d[1:10, ])
  pred_mod_base <- predict(mod_base, type = "response", newdata = d[1:10, ])
  expect_equal(unname(pred_mod), unname(pred_mod_base), tolerance = 1e-2)
})

test_that("feprobit estimation is the same adding noise to the data (stammann centering)", {
  ctrl <- list(centering = "stammann")
  set.seed(123)

  d <- make_probit_data(n = 200, seed = 333)
  d$x1_noisy <- d$x1 + pmax(rnorm(nrow(d)), 0) * .Machine$double.eps

  m1 <- feprobit(y ~ x1 | f1, d, control = ctrl)
  m2 <- feprobit(y ~ x1_noisy | f1, d, control = ctrl)

  expect_equal(unname(coef(m1)), unname(coef(m2)))
  expect_equal(m1$fixed.effects, m2$fixed.effects)
})

test_that("proportional regressors return NA coefficients (stammann centering)", {
  ctrl <- list(centering = "stammann")
  set.seed(200100)
  d <- data.frame(
    y = rbinom(100, 1, 0.5),
    x1 = rnorm(100),
    f = factor(sample(1:2, 100, replace = TRUE))
  )
  d$x2 <- 2 * d$x1

  fit1 <- glm(y ~ x1 + x2 + as.factor(f), data = d, family = binomial(link = "probit"))
  fit2 <- feprobit(y ~ x1 + x2 | f, data = d, control = ctrl)

  expect_equal(coef(fit2), coef(fit1)[2:3], tolerance = 1e-2)
  expect_equal(predict(fit2), predict(fit1, type = "response"), tolerance = 1e-2)
})
