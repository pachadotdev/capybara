test_that("fenegbin is similar to fixest", {
  mod <- fenegbin(mpg ~ wt | cyl | am, data = mtcars)

  # mod_fixest <- fixest::fenegbin(mpg ~ wt | cyl, data = mtcars, cluster = ~am)

  summary_mod <- summary(mod, type = "clustered")
  
  # summary_mod_fixest <- summary(mod_fixest)
  # summary_mod_fixest$coeftable[,2][1]
  summary_mod_fixest <- 0.01889489

  expect_equal(unname(round(summary_mod$cm[,2] - summary_mod_fixest, 1)), 0)
})
