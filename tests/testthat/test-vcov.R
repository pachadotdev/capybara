test_that("vcov works", {
  m1 <- fepoisson(mpg ~ wt + disp | cyl, mtcars)

  m2 <- fepoisson(mpg ~ wt + disp | cyl | carb, mtcars)

  v1 <- vcov(m1)
  v2 <- vcov(m2, type = "clustered")
  v3 <- vcov(m2, type = "sandwich")
  v4 <- vcov(m2, type = "outer.product")

  expect_gt(sum(diag(v1)), sum(diag(v2)))
  expect_gt(sum(diag(v1)), sum(diag(v3)))
  expect_gt(sum(diag(v1)), sum(diag(v4)))
})
