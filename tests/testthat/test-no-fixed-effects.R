#' srr_stats (tests)
#' @srrstats {RE3.1} Validates consistency between `felm`-`feglm` and base R
#' `lm`-`glm` in terms of coefficients when there are no fixed effects of the
#' form `y ~x | f`.
#' @noRd
NULL

# test_that("felm/feglm intercept is ok with no FEs", {
#   skip_on_cran()

#   m1 <- suppressMessages(felm(mpg ~ wt, mtcars))
#   m2 <- suppressMessages(fepoisson(mpg ~ wt, mtcars))

#   f1 <- unname(unlist(fixed_effects(m1)))
#   f2 <- unname(unlist(fixed_effects(m2)))

#   g1 <- unname(coef(lm(mpg ~ wt, mtcars))[1])
#   g2 <- unname(coef(glm(mpg ~ wt, data = mtcars, family = "quasipoisson"))[1])

#   expect_equal(g1, f1, tolerance = 1e-2)
#   expect_equal(g2, f2, tolerance = 1e-2)
# })
