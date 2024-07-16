test_that("felm works", {
  m1 <- felm(mpg ~ wt | cyl, mtcars)
  m2 <- lm(mpg ~ wt + as.factor(cyl), mtcars)

  expect_equal(round(coef(m1), 5), round(coef(m2)[2], 5))
  expect_gt(length(fitted(m1)), 0)
  expect_gt(length(predict(m1)), 0)
  expect_gt(length(coef(m1)), 0)
  expect_gt(length(coef(summary(m1))), 0)
})
