#' srr_stats (tests)
#' @srrstats {G5.2} Validates that covariance matrices align with theoretical expectations under different estimation methods.
#' @srrstats {RE3.3} Ensures consistency of `vcov` results for clustered, sandwich, and outer-product estimators.
#' @srrstats {RE4.3} Confirms robustness of covariance matrix calculations under varied model specifications.
#' @srrstats {RE6.0} Ensures that covariance estimations respond correctly to model clustering and input variations.
#' @noRd
NULL

test_that("vcov works", {
  skip_on_cran()
  
  m1 <- fepoisson(mpg ~ wt + disp | cyl, mtcars)

  m2 <- fepoisson(mpg ~ wt + disp | cyl | carb, mtcars)

  v1 <- vcov(m1)
  v2 <- vcov(m2, type = "clustered")
  v3 <- vcov(m2, type = "sandwich")
  v4 <- vcov(m2, type = "outer.product")

  expect_lte(sum(diag(v1)), sum(diag(v2)))
  expect_lte(sum(diag(v1)), sum(diag(v3)))
  expect_lte(sum(diag(v1)), sum(diag(v4)))
})
