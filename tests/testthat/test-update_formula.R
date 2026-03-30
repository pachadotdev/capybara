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
    feglm(update(fml, . ~ . | cyl), data = mtcars),
    feglm(mpg ~ wt | cyl, data = mtcars)
  )
})
