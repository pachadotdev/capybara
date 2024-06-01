test_that("crossprod_ works", {
  set.seed(123)
  A <- matrix(rnorm(4), nrow = 2)
  w <- c(1,1)

  expect_equal(crossprod(A), crossprod_(A, w, T, T))
  expect_equal(crossprod(A), crossprod_(A, NA_real_, F, F))

  w <- c(1,2)

  # Multiply A by w column-wise
  B <- matrix(NA_real_, nrow = nrow(A), ncol = ncol(A))
  
  for (j in 1:ncol(A)) {
    B[, j] <- A[, j] * w
  }

  expect_equal(crossprod(B), crossprod_(A, w, T, F))

  for (j in 1:ncol(A)) {
    B[, j] <- A[, j] * sqrt(w)
  }

  expect_equal(crossprod(B), crossprod_(A, w, T, T))
})

test_that("backsolve_ works", {
  A <- matrix(c(1, 0, 0, 1, 1, 0, 0, 1, 1), nrow = 3, ncol = 3, byrow = TRUE)
  b <- c(6.50, 7.50, 8.50)

  expect_equal(solve(A,b), solve_beta_(A, b, NA_real_, FALSE))

  # With weights
  # Multiply each column of A by w pair-wise
  # Multiply each b by w pair-wise

  w <- c(1, 2, 3)
  AW <- A * w
  bw <- b * w

  expect_equal(solve(AW, bw), solve_beta_(A, b, w, TRUE))
})
