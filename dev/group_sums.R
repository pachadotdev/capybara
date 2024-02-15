#' @keywords internal
#' @noRd
group_sums_ <- function(M, w, jlist) {
  # Auxiliary variables (fixed)
  J <- length(jlist)
  P <- ncol(M)

  # Initialize the result matrix
  num <- matrix(0, P, 1)

  # Compute sum of weighted group sums
  for (j in 1:J) {
    # Subset j-th group
    indexes <- jlist[[j]]
    I <- length(indexes)

    # Compute numerator of the weighted group sum
    for (p in 1:P) {
      num[p, 1] <- sum(M[indexes, p])
    }

    # Compute denominator of the weighted group sum
    denom <- sum(w[indexes, 1])

    # Divide numerator by denominator
    num[, 1] <- num[, 1] / denom
  }

  # Return vector
  return(num)
}

#' @keywords internal
#' @noRd
group_sums_spectral_ <- function(M, v, w, K, jlist) {
  # Auxiliary variables (fixed)
  J <- length(jlist)
  P <- ncol(M)

  # Initialize the result matrix
  num <- matrix(0, P, 1)

  # Compute sum of weighted group sums
  for (j in 1:J) {
    # Subset j-th group
    indexes <- jlist[[j]]
    I <- length(indexes)

    # Compute numerator of the weighted group sum given bandwidth 'L'
    for (p in 1:P) {
      num[p, 1] <- 0
      for (k in 1:K) {
        for (i in (k + 1):I) {
          num[p, 1] <- num[p, 1] + M[indexes[i], p] * v[indexes[i - k], 1] * I / (I - k)
        }
      }
    }

    # Compute denominator of the weighted group sum
    denom <- sum(w[indexes, 1])

    # Add weighted group sum
    for (p in 1:P) {
      num[p, 1] <- num[p, 1] / denom
    }
  }

  # Return vector
  return(num)
}

#' @keywords internal
#' @noRd
group_sums_var_ <- function(M, jlist) {
  # Auxiliary variables (fixed)
  J <- length(jlist)
  P <- ncol(M)

  # Initialize the result matrix
  V <- matrix(0, P, P)

  # Compute covariance matrix
  for (j in 1:J) {
    # Subset j-th group
    indexes <- jlist[[j]]
    I <- length(indexes)

    # Compute group sum
    v <- rep(0, P)
    for (p in 1:P) {
      v[p] <- sum(M[indexes, p])
    }

    # Add to covariance matrix
    for (p in 1:P) {
      for (q in 1:P) {
        V[p, q] <- V[p, q] + v[p] * v[q]
      }
    }
  }

  # Return matrix
  return(V)
}

#' @keywords internal
#' @noRd
group_sums_cov_ <- function(M, N, jlist) {
  # Auxiliary variables (fixed)
  J <- length(jlist)
  P <- ncol(M)

  # Initialize the result matrix
  V <- matrix(0, P, P)

  # Compute covariance matrix
  for (j in 1:J) {
    # Subset j-th group
    indexes <- jlist[[j]]
    I <- length(indexes)

    # Add to covariance matrix
    for (p in 1:P) {
      for (q in 1:P) {
        for (i in 1:I) {
          for (s in (i + 1):I) {
            V[q, p] <- V[q, p] + M[indexes[i], q] * N[indexes[s], p]
          }
        }
      }
    }
  }

  # Return matrix
  return(V)
}
