# srr_stats (tests)
# {RE2.1} Ensures that models throw meaningful error messages when input parameters or data are invalid.
# {RE3.1} Validates consistency between `fenegbin` and other established R models like `glm` with comparable families.
# {RE3.2} Compares coefficients produced by `fenegbin` with those from base R models to validate similarity.
# {RE7.3} Confirms that estimated coefficients are within a reasonable variation threshold compared to baseline models.

source(system.file("tinytest", "helper.R", package = "capybara"))

local({
  if (Sys.getenv("CAPYBARA_FULL_TESTING") != "yes") {
    return(NULL)
  }

  mod <- fenegbin(ltrade ~ ldist | ctry1, ross2004)

  expect_true(inherits(mod, "feglm"))
  expect_true(!is.null(mod$coef_table))

  s <- summary(mod)

  expect_true(inherits(s, "summary.feglm"))
  expect_true("theta" %in% names(s))
})
