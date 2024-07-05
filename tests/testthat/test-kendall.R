test_that("kendall", {
  x <- 1:2
  expect_equal(kendall_cor(x, x), cor(x, x, method = "kendall"))

  x <- 1:3
  expect_equal(kendall_cor(x, x), 1)

  x <- rep(1, 3)
  expect_equal(kendall_cor(x, x), cor(x, x, method = "kendall"))

  x <- c(1, 0, 2)
  y <- c(5, 3, 4)
  expect_equal(kendall_cor(x, y), cor(x, y, method = "kendall"))

  k1 <- kendall_cor_test(x, y, alternative = "two.sided")
  k2 <- cor.test(x, y, method = "kendall", alternative = "two.sided")
  expect_equal(k1$statistic, unname(k2$estimate))
  expect_equal(k1$p_value, k2$p.value)

  x <- 1:3
  y <- 3:1
  expect_equal(kendall_cor(x, y), cor(x, y, method = "kendall"))

  x <- c(1, NA, 2)
  y <- 3:1
  expect_equal(kendall_cor(x, y), cor(x, y, method = "kendall", use = "pairwise.complete.obs"))

  set.seed(123)
  x <- rnorm(100)
  y <- rpois(100, 2)
  expect_equal(kendall_cor(x, y), cor(x, y, method = "kendall"))

  k1 <- kendall_cor_test(x, y, alternative = "two.sided")
  k2 <- cor.test(x, y, method = "kendall", alternative = "two.sided")
  expect_equal(k1$statistic, unname(k2$estimate))
  expect_equal(k1$p_value, k2$p.value)
  
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

  x <- 1:3
  y <- NA
  expect_error(kendall_cor(x, y))

  x <- 1:3
  y <- c(1, NA, NA)
  expect_error(kendall_cor(x, y))
})
