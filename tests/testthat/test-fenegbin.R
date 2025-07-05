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

  # MASS::glm.nb for negative binomial will return warning because of
  # lack of overdispersion
  # mod_mass <- MASS::glm.nb(
  #   mpg ~ wt + as.factor(cyl),
  #   mtcars
  # )

  mod_base <- glm(
    mpg ~ wt + as.factor(cyl),
    mtcars,
    family = quasipoisson(link = "log")
  )

  expect_equal(coef(mod_base)[2], coef(mod)[1], tolerance = 1e-2)
})
