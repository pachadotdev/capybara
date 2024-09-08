test_that("autoplot works", {
  mod <- felm(mpg ~ wt + qsec | cyl, mtcars)

  expect_s3_class(autoplot(mod, conf_level = 0.99), "ggplot")
  expect_s3_class(autoplot(mod), "ggplot")

  expect_error(autoplot(1L))
  expect_error(autoplot(mod, conf_level = 1.01))
  expect_error(autoplot(mod, conf_level = -0.01))
})
