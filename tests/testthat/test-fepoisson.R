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

test_that("fepoisson is similar to fixest", {
  skip_on_cran()

  # K = 1

  mod <- fepoisson(mpg ~ wt | cyl | am, mtcars)

  mod_base <- glm(
    mpg ~ wt + as.factor(cyl),
    mtcars,
    family = quasipoisson(link = "log")
  )

  # mod
  # fixed_effects(mod)

  # mod_base

  coef_dist_base <- coef(mod_base)[2]

  dist_variation <- abs((coef(mod)[1] - coef_dist_base) / coef(mod)[1])

  expect_gt(0.05, dist_variation)

  expect_output(print(mod))

  expect_visible(summary(mod, type = "cluster"))

  fes <- fixed_effects(mod)
  n <- unname(mod[["nobs"]]["nobs_full"])
  expect_equal(length(fes), 1)
  expect_equal(length(fitted(mod)), n)
  expect_equal(length(predict(mod)), n)
  expect_equal(length(coef(mod)), 1)
  expect_equal(length(fes), 1)

  expect_equal(
    unname(fes[["cyl"]][1]),
    unname(coef(glm(mpg ~ wt + as.factor(cyl), mtcars, family = quasipoisson(link = "log")))[1]),
    tolerance = 1e-2
  )

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
})

test_that("fepoisson estimation is the same adding noise to the data", {
  set.seed(123)
  d <- data.frame(
    x = rnorm(1000),
    y = rpois(1000, 1),
    f = factor(rep(1:10, 100))
  )

  set.seed(123)
  d$y2 <- d$y + pmax(rnorm(nrow(d)), 0) * .Machine$double.eps

  m1 <- fepoisson(y ~ x | f, d)
  m2 <- fepoisson(y2 ~ x | f, d)
  expect_equal(coef(m1), coef(m2))
  expect_equal(fixed_effects(m1), fixed_effects(m2))
})
