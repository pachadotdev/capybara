test_that("crossprod works", {
  A <- matrix(c(1, 0, 0, 1), nrow = 2, ncol = 2)
  expect_equal(crossprod(A), crossprod_(A, NA_real_, FALSE, FALSE))

  b <- c(1, 2)
  expect_equal(crossprod(A * b), crossprod_(A, b, TRUE, FALSE))
  expect_equal(crossprod(A * sqrt(b)), crossprod_(A, b, TRUE, TRUE))
})

test_that("solve works", {
  A <- matrix(c(1, 0, 0, 1), nrow = 2, ncol = 2)
  b <- c(2, 2)
  expect_equal(solve(A, b), solve_(A, b))
})

test_that("chol works", {
  set.seed(123)
  A <- matrix(c(5, 1, 1, 3), 2, 2)
  expect_equal(chol(A), chol_(A))
})

test_that("chol2inv works", {
  cma <- chol(ma <- cbind(1, 1:3, c(1, 3, 7)))
  expect_equal(ma %*% chol2inv(cma), ma %*% chol2inv_(cma))
})
