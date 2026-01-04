#' srr_stats (tests)
#' @srrstats {RE3.1} Validates consistency between `fepoisson` and other established R models like `glm` with comparable families.
#' @srrstats {RE3.2} Compares coefficients produced by `fepoisson` with those from base R models to validate similarity.
#' @srrstats {RE4.3} Ensures stable estimates when adding negligible noise to the data.
#' @srrstats {RE5.1} Validates proper output generation for the model summary and printing methods.
#' @srrstats {RE7.3} Confirms that estimated coefficients are within a reasonable variation threshold compared to baseline models.
#' @srrstats {G5.10} The CAPYBARA_EXTENDED_TESTS environment variable can be set to true to run extended tests.
#' @srrstats {G5.11} The extended tests do not require additional downloads.
#' @srrstats {G5.11a} As for G5.11., the extended tests do not require additional downloads.
#' @srrstats {G5.12} The extended tests verify that the algorithm fitting time is robust to noise. This has to be tested with a larger dataset to see that time(clean) <= time(noisy).
#' @noRd
NULL

test_that("fepoisson is similar to base", {
  skip_on_cran()

  # K = 1

  mod <- fepoisson(mpg ~ wt | cyl | am, mtcars)

  mod_base <- glm(
    mpg ~ wt + as.factor(cyl),
    mtcars,
    family = quasipoisson(link = "log")
  )

  coef_dist_base <- coef(mod_base)[2]

  dist_variation <- unname(abs((coef(mod)[1] - coef_dist_base) / coef(mod)[1]))

  expect_equal(dist_variation, 0.0, tolerance = 1e-2)

  expect_output(print(mod))

  expect_visible(summary(mod))

  # fes <- fixed_effects(mod)
  n <- unname(mod[["nobs"]]["nobs_full"])
  # expect_equal(length(fes), 1)
  expect_equal(length(fitted(mod)), n)
  expect_equal(length(predict(mod)), n)
  expect_equal(length(coef(mod)), 1)

  # expect_equal(
  #   unname(fes[["cyl"]][1]),
  #   unname(coef(glm(mpg ~ wt + as.factor(cyl), mtcars, family = quasipoisson(link = "log")))[1]),
  #   tolerance = 1e-2
  # )

  smod <- summary(mod)

  expect_equal(length(coef(smod)[, 1]), 1)
  expect_output(summary_formula_(smod))
  expect_output(summary_family_(smod))
  expect_output(summary_estimates_(smod, 3))
  expect_output(summary_r2_(smod, 3))
  expect_output(summary_nobs_(smod))
  expect_output(summary_fisher_(smod))

  # K = 2

  mod <- fepoisson(mpg ~ wt | cyl + am, mtcars)

  mod_base <- glm(
    mpg ~ wt + as.factor(cyl) + as.factor(am),
    mtcars,
    family = quasipoisson(link = "log")
  )

  # mod$coefficients
  # mod_base$coefficients
  
  coef_dist_base <- coef(mod_base)[2]

  dist_variation <- abs((coef(mod)[1] - coef_dist_base) / coef(mod)[1])

  expect_lt(dist_variation, 0.05)

  # K = 3

  mod <- fepoisson(mpg ~ wt | cyl + am + carb, mtcars)

  mod_base <- glm(
    mpg ~ wt + as.factor(cyl) + as.factor(am) + as.factor(carb),
    mtcars,
    family = quasipoisson(link = "log")
  )

  coef_dist_base <- coef(mod_base)[2]

  dist_variation <- abs((coef(mod)[1] - coef_dist_base) / coef(mod)[1])

  expect_lt(dist_variation, 0.05)

  # mod$coefficients
  # mod_base$coefficients

  expect_equal(mod[["fitted_values"]], mod_base[["fitted.values"]], tolerance = 1e-2)

  pred_mod <- predict(mod, type = "response")
  pred_mod_base <- predict(mod_base, type = "response")

  pred_mod_link <- predict(mod, type = "link")
  pred_mod_base_link <- predict(mod_base, type = "link")

  expect_equal(pred_mod, pred_mod_base, tolerance = 1e-2)
  expect_equal(pred_mod_link, pred_mod_base_link, tolerance = 1e-2)

  pred_mod <- predict(mod, type = "response", newdata = mtcars[1:10, ])
  pred_mod_base <- predict(mod_base, type = "response", newdata = mtcars[1:10, ])

  pred_mod_link <- predict(mod, type = "link", newdata = mtcars[1:10, ])
  pred_mod_base_link <- predict(mod_base, type = "link", newdata = mtcars[1:10, ])

  expect_equal(pred_mod, pred_mod_base, tolerance = 1e-2)
  expect_equal(pred_mod_link, pred_mod_base_link, tolerance = 1e-2)
})

test_that("fepoisson estimation is the same adding noise to the data", {
  set.seed(123)
  d <- mtcars[, c("mpg", "wt", "cyl")]
  d$wt2 <- d$wt + pmax(rnorm(nrow(d)), 0) * .Machine$double.eps

  m1 <- fepoisson(mpg ~ wt | cyl, d)
  m2 <- fepoisson(mpg ~ wt2 | cyl, d)

  expect_equal(unname(coef(m1)), unname(coef(m2)))
  expect_equal(m1$fixed.effects, m2$fixed.effects)
})


test_that("proportional regressors return NA coefficients", {
  set.seed(200100)
  d <- data.frame(
    y = rpois(100, 2),
    x1 = rnorm(100),
    f = factor(sample(1:2, 100, replace = TRUE))
  )
  d$x2 <- 2 * d$x1

  fit1 <- glm(y ~ x1 + x2 + as.factor(f), data = d, family = poisson())
  fit2 <- feglm(y ~ x1 + x2 | f, data = d, family = poisson())

  expect_equal(coef(fit2), coef(fit1)[2:3], tolerance = 1e-2)
  expect_equal(predict(fit2), predict(fit1), tolerance = 1e-2)
})
