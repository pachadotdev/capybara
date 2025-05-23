#' srr_stats (tests)
#' @srrstats {RE2.1} Ensures that models throw meaningful error messages when input parameters or data are invalid.
#' @srrstats {RE5.1} Validates appropriate error handling for omitted arguments, such as missing formula or data.
#' @noRd
NULL

test_that("felm with weights works", {
  skip_on_cran()
  
  m1 <- felm(mpg ~ wt | am, weights = ~cyl, data = mtcars)
  m2 <- felm(mpg ~ wt | am, weights = mtcars$cyl, data = mtcars)

  w <- mtcars$cyl
  m3 <- felm(mpg ~ wt | am, weights = w, data = mtcars)

  expect_equal(coef(m1), coef(m2))
  expect_equal(coef(m1), coef(m3))

  w <- NULL
  m4 <- felm(mpg ~ wt | am, weights = w, data = mtcars)

  expect_gt(coef(m1), coef(m4))
})

test_that("feglm with weights works", {
  m1 <- feglm(mpg ~ wt | am, weights = ~cyl, data = mtcars)
  m2 <- feglm(mpg ~ wt | am, weights = mtcars$cyl, data = mtcars)

  w <- mtcars$cyl
  m3 <- feglm(mpg ~ wt | am, weights = w, data = mtcars)

  expect_equal(coef(m1), coef(m2))
  expect_equal(coef(m1), coef(m3))

  w <- NULL
  m4 <- feglm(mpg ~ wt | am, weights = w, data = mtcars)

  expect_gt(coef(m1), coef(m4))
})
