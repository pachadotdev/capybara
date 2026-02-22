#' srr_stats (tests)
#' @srrstats {RE2.1} Ensures that models throw meaningful error messages when input parameters or data are invalid.
#' @srrstats {RE3.1} Validates consistency between `fenegbin` and other established R models like `glm` with comparable families.
#' @srrstats {RE3.2} Compares coefficients produced by `fenegbin` with those from base R models to validate similarity.
#' @srrstats {RE7.3} Confirms that estimated coefficients are within a reasonable variation threshold compared to baseline models.
#' @noRd
NULL

test_that("fenegbin is similar to fixest", {
  skip_on_cran()

  mod <- fenegbin(mpg ~ wt | cyl, mtcars)
  # fepoisson(mpg ~ wt | cyl, mtcars)

  # MASS::glm.nb for negative binomial will return warning because of
  # lack of overdispersion
  mod_mass <- suppressWarnings(MASS::glm.nb(
    mpg ~ wt + as.factor(cyl),
    mtcars
  ))

  expect_equal(coef(mod)[1], coef(mod_mass)[2], tolerance = 0.05)
})

test_that("fenegbin returns correct structure", {
  skip_on_cran()

  mod <- fenegbin(mpg ~ wt | cyl, mtcars)

  expect_s3_class(mod, "feglm")
  expect_true("theta" %in% names(mod))
  expect_true("coef_table" %in% names(mod))
  expect_true("deviance" %in% names(mod))
  expect_true("null_deviance" %in% names(mod))
})

test_that("fenegbin works with multiple predictors", {
  skip_on_cran()

  mod <- fenegbin(mpg ~ wt + hp + disp | cyl, mtcars)

  expect_equal(length(coef(mod)), 3)
  expect_true(all(is.finite(coef(mod))))
})

test_that("fenegbin works with multiple fixed effects", {
  skip_on_cran()

  mod <- fenegbin(mpg ~ wt | cyl + am, mtcars)

  expect_s3_class(mod, "feglm")
  expect_true("theta" %in% names(mod))
})

test_that("fenegbin theta parameter is positive", {
  skip_on_cran()

  mod <- fenegbin(mpg ~ wt | cyl, mtcars)

  expect_true(mod$theta > 0)
  expect_true(is.finite(mod$theta))
})

test_that("fenegbin works with clustering", {
  skip_on_cran()

  mod <- fenegbin(mpg ~ wt | cyl | carb, mtcars)

  expect_s3_class(mod, "feglm")
  expect_true(!is.null(mod$vcov))
})

test_that("fenegbin completes fitting", {
  skip_on_cran()

  mod <- fenegbin(mpg ~ wt | cyl, mtcars)

  expect_s3_class(mod, "feglm")
  expect_true(!is.null(mod$coef_table))
})

test_that("fenegbin summary works", {
  skip_on_cran()

  mod <- fenegbin(mpg ~ wt | cyl, mtcars)
  s <- summary(mod)

  expect_s3_class(s, "summary.feglm")
  expect_true("theta" %in% names(s))
})

test_that("fenegbin works with different data", {
  skip_on_cran()

  # Test with a different dataset
  data("trade_panel", package = "capybara")
  mod <- fenegbin(trade ~ log_dist | exp_year, trade_panel)

  expect_s3_class(mod, "feglm")
  expect_true(!is.null(mod$coef_table))
})

test_that("fenegbin respects control parameters", {
  skip_on_cran()

  ctrl <- fit_control(dev_tol = 1e-10, iter_max = 50L)
  mod <- fenegbin(mpg ~ wt | cyl, mtcars, control = ctrl)

  expect_s3_class(mod, "feglm")
  expect_true(!is.null(mod$coef_table))
})

# Stammann centering ----

test_that("fenegbin is similar to fixest (stammann centering)", {
  skip_on_cran()
  ctrl <- list(centering = "stammann")

  mod <- fenegbin(mpg ~ wt | cyl, mtcars, control = ctrl)
  mod_mass <- suppressWarnings(MASS::glm.nb(
    mpg ~ wt + as.factor(cyl),
    mtcars
  ))

  expect_equal(coef(mod)[1], coef(mod_mass)[2], tolerance = 0.05)
})

test_that("fenegbin returns correct structure (stammann centering)", {
  skip_on_cran()
  ctrl <- list(centering = "stammann")

  mod <- fenegbin(mpg ~ wt | cyl, mtcars, control = ctrl)

  expect_s3_class(mod, "feglm")
  expect_true("theta" %in% names(mod))
  expect_true("coef_table" %in% names(mod))
  expect_true("deviance" %in% names(mod))
  expect_true("null_deviance" %in% names(mod))
})

test_that("fenegbin works with multiple fixed effects (stammann centering)", {
  skip_on_cran()
  ctrl <- list(centering = "stammann")

  mod <- fenegbin(mpg ~ wt | cyl + am, mtcars, control = ctrl)

  expect_s3_class(mod, "feglm")
  expect_true("theta" %in% names(mod))
})

test_that("fenegbin theta is positive (stammann centering)", {
  skip_on_cran()
  ctrl <- list(centering = "stammann")

  mod <- fenegbin(mpg ~ wt | cyl, mtcars, control = ctrl)

  expect_true(mod$theta > 0)
  expect_true(is.finite(mod$theta))
})

test_that("fenegbin respects control parameters (stammann centering)", {
  skip_on_cran()

  ctrl <- fit_control(dev_tol = 1e-10, iter_max = 50L, centering = "stammann")
  mod <- fenegbin(mpg ~ wt | cyl, mtcars, control = ctrl)

  expect_s3_class(mod, "feglm")
  expect_true(!is.null(mod$coef_table))
})
