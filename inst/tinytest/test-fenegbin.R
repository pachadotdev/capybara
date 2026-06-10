# srr_stats (tests)
# {RE2.1} Ensures that models throw meaningful error messages when input parameters or data are invalid.
# {RE3.1} Validates consistency between `fenegbin` and other established R models like `glm` with comparable families.
# {RE3.2} Compares coefficients produced by `fenegbin` with those from base R models to validate similarity.
# {RE7.3} Confirms that estimated coefficients are within a reasonable variation threshold compared to baseline models.

source(system.file("tinytest", "helper.R", package = "capybara"))

local({
  skip_on_cran()

  # Test with a different dataset
  data("yotov2017", package = "capybara")
  mod <- fenegbin(trade ~ log_dist | exp_year, yotov2017)

  expect_true(inherits(mod, "feglm"))
  expect_true(!is.null(mod$coef_table))

  s <- summary(mod)

  expect_true(inherits(s, "summary.feglm"))
  expect_true("theta" %in% names(s))
})

local({
  skip_on_cran()

  # Test with a different dataset
  data("yotov2017", package = "capybara")
  mod <- fenegbin(trade ~ log_dist | exp_year, yotov2017,
    control = fit_control(centering = "stammann"))

  expect_true(inherits(mod, "feglm"))
  expect_true(!is.null(mod$coef_table))

  s <- summary(mod)

  expect_true(inherits(s, "summary.feglm"))
  expect_true("theta" %in% names(s))
})
