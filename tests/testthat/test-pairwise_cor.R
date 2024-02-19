test_that("cor(x,x) = 1", {
  x <- c(1, 2, 3, 4, 5)
  expect_equal(pairwise_cor_(x, x), 1)
  expect_equal(pairwise_cor_(x, -x), -1)
})
