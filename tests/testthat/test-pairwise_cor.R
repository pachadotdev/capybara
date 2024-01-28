test_that("cor(x,x) = 1", {
  expect_equal(pairwise_cor_(x, x), 1)
})
