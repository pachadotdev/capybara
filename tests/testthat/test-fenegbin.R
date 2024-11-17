#' srr_stats (tests)
#' @srrstatsVerbose TRUE
#' @srrstats {G5.4b} See test-feglm.R
#' @noRd
NULL

test_that("fenegbin is similar to fixest", {
  mod <- fenegbin(mpg ~ wt | cyl, mtcars)

  mod_base <- glm(
    mpg ~ wt + as.factor(cyl),
    mtcars,
    family = quasipoisson(link = "log")
  )

  coef_dist_base <- coef(mod_base)[2]

  dist_variation <- abs((coef(mod)[1] - coef_dist_base) / coef(mod)[1])

  expect_lt(dist_variation, 0.05)
})
