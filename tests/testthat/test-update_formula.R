test_that("felm + updated fixed effects in formula", {
  fml <- mpg ~ wt | am
  mod1 <- felm(update(fml, . ~ . | cyl), data = mtcars)
  mod2 <- felm(mpg ~ wt | cyl, data = mtcars)
  # Compare coefficients with tolerance for cross-platform numerical stability
  expect_equal(coef(mod1), coef(mod2), tolerance = 1e-8)
  expect_equal(fitted(mod1), fitted(mod2), tolerance = 1e-8)
})

test_that("feglm + updated fixed effects in formula", {
  fml <- mpg ~ wt | am
  mod1 <- feglm(update(fml, . ~ . | cyl), data = mtcars)
  mod2 <- feglm(mpg ~ wt | cyl, data = mtcars)
  # Compare coefficients with tolerance for cross-platform numerical stability
  expect_equal(coef(mod1), coef(mod2), tolerance = 1e-8)
  expect_equal(fitted(mod1), fitted(mod2), tolerance = 1e-8)
})
