#' Group sums
#' @param M matrix
#' @param w vector
#' @param jlist list
groupSums <- function(M, w, jlist) {
  # Input validation
  # if (!is.matrix(M)) stop("M must be a matrix")
  # if (!is.numeric(w)) stop("w must be a numeric vector")
  # if (!is.list(jlist)) stop("jlist must be a list of integer vectors")

  # Auxiliary variables (fixed)
  J <- length(jlist)
  P <- ncol(M)

  # Compute sum of weighted group sums
  b <- vapply(seq_len(P), function(p) {
    sum(vapply(seq_len(J), function(j) {
      # Subset j-th group
      indexes <- jlist[[j]]

      # Compute numerator of the weighted group sum
      num <- sum(M[indexes, p])

      # Compute denominator of the weighted group sum
      denom <- sum(w[indexes])

      # Avoid division by zero
      # if (denom == 0) denom <- .Machine$double.eps

      # Return weighted group sum
      # ifelse(denom == 0, Inf, num / denom)
      return(num / denom)
    }, numeric(length(J))))
  }, numeric(length(P)))

  # Return vector
  return(b)
}

#' Group sums with spectral weights
#' @param M matrix
#' @param v vector
#' @param w vector
#' @param L integer
#' @param jlist list
groupSumsSpectral <- function(M, v, w, L, jlist) {
  # Input validation
  # if (!is.matrix(M)) stop("M must be a matrix")
  # if (!is.numeric(v)) stop("v must be a numeric vector")
  # if (!is.numeric(w)) stop("w must be a numeric vector")
  # if (!is.integer(L) || L <= 0) stop("L must be a positive integer")
  # if (!is.list(jlist)) stop("jlist must be a list of integer vectors")

  # Auxiliary variables (fixed)
  J <- length(jlist)
  P <- ncol(M)

  # Compute sum of weighted group sums
  b <- vapply(seq_len(P), function(p) {
    sum(vapply(seq_len(J), function(j) {
      # Subset j-th group
      indexes <- jlist[[j]]
      K <- length(indexes)

      # Compute numerator of the weighted group sum given bandwidth 'L'
      num <- sum(vapply(seq(from = 1, to = min(L, K)), function(l) {
        sum(M[indexes[l:K], p] * v[indexes[1:(K - l + 1)]] * K / (K - l))
      }, numeric(1)))

      # Compute denominator of the weighted group sum
      denom <- sum(w[indexes])

      # Avoid division by zero
      # if (denom == 0) denom <- .Machine$double.eps

      # Return weighted group sum
      # ifelse(denom == 0, Inf, num / denom)
      return(num / denom)
    }, numeric(1)))
  }, numeric(P))

  # Return vector
  return(b)
}

#' Group sums variance
#' @param M matrix
#' @param jlist list
groupSumsVar <- function(M, jlist) {
  # Input validation
  # if (!is.matrix(M)) stop("M must be a matrix")
  # if (!is.list(jlist)) stop("jlist must be a list of integer vectors")

  # Auxiliary variables (fixed)
  J <- length(jlist)
  P <- ncol(M)

  # Auxiliary variables (storage)
  V <- matrix(0, P, P)

  # Compute covariance matrix
  for (j in seq_len(J)) {
    # Subset j-th group
    indexes <- jlist[[j]]

    # Compute group sum
    v <- colSums(M[indexes, ])

    # Add to covariance matrix
    V <- V + v %*% t(v)
  }

  # Return matrix
  return(V)
}

#' Group sums covariance
#' @param M matrix
#' @param N matrix
#' @param jlist list
groupSumsCov <- function(M, N, jlist) {
  # Input validation
  # if (!is.matrix(M)) stop("M must be a matrix")
  # if (!is.matrix(N)) stop("N must be a matrix")
  # if (!is.list(jlist)) stop("jlist must be a list of integer vectors")

  P <- ncol(M)
  V <- matrix(0, nrow = P, ncol = P)

  for (indexes in jlist) {
    # Update the covariance matrix using matrix multiplication
    V <- V + t(M[indexes, ]) %*% N[indexes, ]
  }

  return(V)
}
