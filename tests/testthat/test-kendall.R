test_that("kendall", {
  x <- 1:3
  expect_equal(kendall_cor(x, x), 1)

  x <- c(1, 0, 2)
  y <- c(5, 3, 4)
  expect_equal(kendall_cor(x, y), cor(x, y, method = "kendall"))

  x <- 1:3
  y <- 3:1
  expect_equal(kendall_cor(x, y), cor(x, y, method = "kendall"))

  set.seed(123)
  x <- rnorm(100)
  y <- rpois(100, 2)
  expect_equal(kendall_cor(x, y), cor(x, y, method = "kendall"))

  x <- rnorm(1e3)
  y <- rpois(1e3, 2)
  expect_equal(kendall_cor(x, y), cor(x, y, method = "kendall"))

  t_kendall <- c()
  for (i in 1:100) {
    t1 <- Sys.time()
    kendall_cor(x, y)
    t2 <- Sys.time()
    t_kendall[i] <- t2 - t1
  }
  t_kendall <- median(t_kendall)


  t_cor <- c()
  for (i in 1:100) {
    t1 <- Sys.time()
    cor(x, y, method = "kendall")
    t2 <- Sys.time()
    t_cor[i] <- t2 - t1
  }
  t_cor <- median(t_cor)

  expect_lt(t_kendall, t_cor)
})
