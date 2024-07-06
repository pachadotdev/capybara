test_that("crossprod works", {
  A <- matrix(c(1, 0, 0, 1), nrow = 2, ncol = 2)
  expect_equal(crossprod(A), crossprod_(A, NA_real_, FALSE, FALSE))

  b <- c(1, 2)
  expect_equal(crossprod(A * b), crossprod_(A, b, TRUE, FALSE))
  expect_equal(crossprod(A * sqrt(b)), crossprod_(A, b, TRUE, TRUE))
})

test_that("solve_bias_ works", {
  A <- matrix(c(1, 0, 0, 1), nrow = 2, ncol = 2)
  x <- c(2, 2)
  expect_equal(as.vector(A %*% x), solve_y_(A, x))
  expect_equal(x - solve(A, x), solve_bias_(x, A, 1, x))
})

test_that("inv_ works", {
  A <- matrix(c(1, 0, 0, 1, 1, 0, 0, 1, 1), nrow = 3, ncol = 3, byrow = TRUE)
  expect_equal(solve(A), inv_(A))

  # non-invertible matrix
  A <- matrix(c(1, 0, 0, 1, 0, 0, 0, 1, 1), nrow = 3, ncol = 3, byrow = TRUE)
  expect_error(inv_(A))

  # non-square matrix
  A <- matrix(c(1, 0, 0, 1, 1, 0), nrow = 2, ncol = 3, byrow = TRUE)
  expect_error(inv_(A))
})
