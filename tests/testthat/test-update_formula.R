test_that("felm + updated fixed effects in formula", {
  fml <- mpg ~ wt | am
  expect_equal(
    felm(update(fml, . ~ . | cyl), data = mtcars),
    felm(mpg ~ wt | cyl, data = mtcars)
  )
})

test_that("feglm + updated fixed effects in formula", {
  fml <- mpg ~ wt | am
  expect_equal(
    # TODO: using coef() for strange Mac issue (check later)
    coef(feglm(update(fml, . ~ . | cyl), data = mtcars)),
    coef(feglm(mpg ~ wt | cyl, data = mtcars))
  )
})
