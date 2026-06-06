#' srr_stats (tests)
#' @srrstats {RE3.1} Validates consistency between `fepoisson` and other established R models like `glm` with comparable families.
#' @srrstats {RE3.2} Compares coefficients produced by `fepoisson` with those from base R models to validate similarity.
#' @srrstats {RE4.3} Ensures stable estimates when adding negligible noise to the data.
#' @srrstats {RE5.1} Validates proper output generation for the model summary and printing methods.
#' @srrstats {RE7.3} Confirms that estimated coefficients are within a reasonable variation threshold compared to baseline models.
#' @srrstats {G5.11} The extended tests do not require additional downloads.
#' @srrstats {G5.11a} As for G5.11., the extended tests do not require additional downloads.
#' @srrstats {G5.12} The extended tests verify that the algorithm fitting time is robust to noise. This has to be tested with a larger dataset to see that time(clean) <= time(noisy).
#' @noRd
NULL

source(system.file("tinytest", "helper.R", package = "capybara"))

# fepoisson_asymmetric is similar to fepoisson at 50% expectile"
local({
  skip_on_cran()

  mod1 <- fepoisson_asymmetric(mpg ~ wt | cyl | am, mtcars, control = fit_control(expectile = 0.5, return_fe = TRUE))

  mod2 <- fepoisson(mpg ~ wt | cyl | am, mtcars, control = fit_control(return_fe = TRUE))

  expect_equal(coef(mod1), coef(mod2), tolerance = 1e-2)
  expect_equal(mod1$fixed_effects, mod2$fixed_effects, tolerance = 1e-2)
  expect_equal(fitted(mod1), fitted(mod2), tolerance = 1e-2)
})

# fepoisson_asymmetric slopes are smaller than fepoisson at 25% expectile"
local({
  skip_on_cran()

  mod1 <- fepoisson_asymmetric(mpg ~ wt | cyl | am, mtcars, control = fit_control(expectile = 0.25, return_fe = TRUE))

  mod2 <- fepoisson(mpg ~ wt | cyl | am, mtcars, control = fit_control(return_fe = TRUE))

  expect_true(coef(mod1) < coef(mod2))
})

# fepoisson_asymmetric with expectile_glm_iter_max = 1L gives same result as default"
local({
  skip_on_cran()

  mod1 <- fepoisson_asymmetric(
    mpg ~ wt | cyl | am,
    mtcars,
    control = fit_control(expectile = 0.25, expectile_iter_max = 500L)
  )

  mod2 <- fepoisson_asymmetric(
    mpg ~ wt | cyl | am,
    mtcars,
    control = fit_control(expectile = 0.25, expectile_glm_iter_max = 1L, expectile_iter_max = 500L)
  )

  expect_equal(coef(mod1), coef(mod2), tolerance = 1e-4)
})
