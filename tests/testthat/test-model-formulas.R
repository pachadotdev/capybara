test_that("formulas with operators are handled correctly", {
  # log-log
  m1 <- felm(log(mpg) ~ log(wt) | cyl, mtcars)
  m2 <- lm(log(mpg) ~ log(wt) + as.factor(cyl), mtcars)
  expect_equal(coef(m1), coef(m2)[2], tolerance = 1e-2)

  # log-linear
  m1 <- felm(log(mpg) ~ wt | cyl, mtcars)
  m2 <- lm(log(mpg) ~ wt + as.factor(cyl), mtcars)
  expect_equal(coef(m1), coef(m2)[2], tolerance = 1e-2)

  # linear-log
  m1 <- felm(mpg ~ log(wt) | cyl, mtcars)
  m2 <- lm(mpg ~ log(wt) + as.factor(cyl), mtcars)
  expect_equal(coef(m1), coef(m2)[2], tolerance = 1e-2)

  # other operators
  m1 <- felm(mpg ~ I(wt^2) + qsec | cyl, mtcars)
  m2 <- lm(mpg ~ I(wt^2) + qsec + as.factor(cyl), mtcars)
  expect_equal(coef(m1), coef(m2)[2:3], tolerance = 1e-2)

  # interaction terms
  m1 <- felm(mpg ~ wt * qsec | cyl, mtcars)
  m2 <- lm(mpg ~ wt * qsec + as.factor(cyl), mtcars)
  expect_equal(coef(m1), coef(m2)[c(2, 3, 6)], tolerance = 1e-2)
})
