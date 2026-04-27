test_that("formulas with operators are handled correctly", {
  # log-log
  expect_error(felm(log(mpg) ~ log(wt) | cyl, mtcars))

  # log-linear
  expect_error(felm(log(mpg) ~ wt | cyl, mtcars))

  # linear-log
  expect_error(felm(mpg ~ log(wt) | cyl, mtcars))

  # identity operator
  expect_error(felm(mpg ~ I(wt^2) + qsec | cyl, mtcars))

  # polynomial terms
  expect_error(felm(mpg ~ poly(wt, 2) | cyl, mtcars))

  # interaction terms
  m1 <- felm(mpg ~ wt * qsec | cyl, mtcars)
  m2 <- lm(mpg ~ wt * qsec + as.factor(cyl), mtcars)
  expect_equal(coef(m1), coef(m2)[c(2, 3, 6)], tolerance = 1e-2)
})
