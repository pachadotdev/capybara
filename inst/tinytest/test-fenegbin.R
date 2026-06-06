#' srr_stats (tests)
#' @srrstats {RE2.1} Ensures that models throw meaningful error messages when input parameters or data are invalid.
#' @srrstats {RE3.1} Validates consistency between `fenegbin` and other established R models like `glm` with comparable families.
#' @srrstats {RE3.2} Compares coefficients produced by `fenegbin` with those from base R models to validate similarity.
#' @srrstats {RE7.3} Confirms that estimated coefficients are within a reasonable variation threshold compared to baseline models.
#' @noRd
NULL

source(system.file("tinytest", "helper.R", package = "capybara"))

# fenegbin is similar to fixest"
local({
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

# fenegbin returns correct structure"
local({
  skip_on_cran()

  mod <- fenegbin(mpg ~ wt | cyl, mtcars)

  expect_true(inherits(mod, "feglm"))
  expect_true("theta" %in% names(mod))
  expect_true("coef_table" %in% names(mod))
  expect_true("deviance" %in% names(mod))
  expect_true("null_deviance" %in% names(mod))
})

# fenegbin works with multiple predictors"
local({
  skip_on_cran()

  mod <- fenegbin(mpg ~ wt + hp + disp | cyl, mtcars)

  expect_equal(length(coef(mod)), 3)
  expect_true(all(is.finite(coef(mod))))
})

# fenegbin works with multiple fixed effects"
local({
  skip_on_cran()

  mod <- fenegbin(mpg ~ wt | cyl + am, mtcars)

  expect_true(inherits(mod, "feglm"))
  expect_true("theta" %in% names(mod))
})

# fenegbin theta parameter is positive"
local({
  skip_on_cran()

  mod <- fenegbin(mpg ~ wt | cyl, mtcars)

  expect_true(mod$theta > 0)
  expect_true(is.finite(mod$theta))
})

# fenegbin works with clustering"
local({
  skip_on_cran()

  mod <- fenegbin(mpg ~ wt | cyl | carb, mtcars)

  expect_true(inherits(mod, "feglm"))
  expect_true(!is.null(mod$vcov))
})

# fenegbin completes fitting"
local({
  skip_on_cran()

  mod <- fenegbin(mpg ~ wt | cyl, mtcars)

  expect_true(inherits(mod, "feglm"))
  expect_true(!is.null(mod$coef_table))
})

# fenegbin summary works"
local({
  skip_on_cran()

  mod <- fenegbin(mpg ~ wt | cyl, mtcars)
  s <- summary(mod)

  expect_true(inherits(s, "summary.feglm"))
  expect_true("theta" %in% names(s))
})

# fenegbin works with different data"
local({
  skip_on_cran()

  # Test with a different dataset
  data("yotov2017", package = "capybara")
  mod <- fenegbin(trade ~ log_dist | exp_year, yotov2017)

  expect_true(inherits(mod, "feglm"))
  expect_true(!is.null(mod$coef_table))
})

# fenegbin respects control parameters"
local({
  skip_on_cran()

  ctrl <- fit_control(dev_tol = 1e-10, iter_max = 50L)
  mod <- fenegbin(mpg ~ wt | cyl, mtcars, control = ctrl)

  expect_true(inherits(mod, "feglm"))
  expect_true(!is.null(mod$coef_table))
})

# Stammann centering ----

# fenegbin is similar to fixest (stammann centering)"
local({
  skip_on_cran()
  ctrl <- list(centering = "stammann")

  mod <- fenegbin(mpg ~ wt | cyl, mtcars, control = ctrl)
  mod_mass <- suppressWarnings(MASS::glm.nb(
    mpg ~ wt + as.factor(cyl),
    mtcars
  ))

  expect_equal(coef(mod)[1], coef(mod_mass)[2], tolerance = 0.05)
})

# fenegbin returns correct structure (stammann centering)"
local({
  skip_on_cran()
  ctrl <- list(centering = "stammann")

  mod <- fenegbin(mpg ~ wt | cyl, mtcars, control = ctrl)

  expect_true(inherits(mod, "feglm"))
  expect_true("theta" %in% names(mod))
  expect_true("coef_table" %in% names(mod))
  expect_true("deviance" %in% names(mod))
  expect_true("null_deviance" %in% names(mod))
})

# fenegbin works with multiple fixed effects (stammann centering)"
local({
  skip_on_cran()
  ctrl <- list(centering = "stammann")

  mod <- fenegbin(mpg ~ wt | cyl + am, mtcars, control = ctrl)

  expect_true(inherits(mod, "feglm"))
  expect_true("theta" %in% names(mod))
})

# fenegbin theta is positive (stammann centering)"
local({
  skip_on_cran()
  ctrl <- list(centering = "stammann")

  mod <- fenegbin(mpg ~ wt | cyl, mtcars, control = ctrl)

  expect_true(mod$theta > 0)
  expect_true(is.finite(mod$theta))
})

# fenegbin respects control parameters (stammann centering)"
local({
  skip_on_cran()

  ctrl <- fit_control(dev_tol = 1e-10, iter_max = 50L, centering = "stammann")
  mod <- fenegbin(mpg ~ wt | cyl, mtcars, control = ctrl)

  expect_true(inherits(mod, "feglm"))
  expect_true(!is.null(mod$coef_table))
})
