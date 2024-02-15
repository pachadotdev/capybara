#' @keywords internal
#' @noRd
get_alpha_ <- function(p, klist, tol) {
  # Auxiliary variables (fixed)
  N <- nrow(p)
  K <- length(klist)

  # Generate starting guess
  Alpha <- vector("list", K)
  for (k in 1:K) {
    Alpha[[k]] <- matrix(0, length(klist[[k]]), 1)
  }

  # Start alternating between normal equations
  for (iter in 1:10000) {
    # Store \alpha_{0} of the previous iteration
    Alpha0 <- Alpha

    # Solve normal equations of category k
    for (k in seq_len(K)) {
      # Compute adjusted dependent variable
      y <- p

      for (l in 1:K) {
        if (l != k) {
          for (j in seq_len(klist[[l]])) {
            for (i in seq_len(jlist[[j]])) {
              y[jlist[[j]][i], 1] <- y[jlist[[j]][i], 1] - Alpha[[l]][j, 1]
            }
          }
        }
      }

      # Compute group mean
      jlist <- klist[[k]]
      alpha <- matrix(0, length(jlist), 1)

      # Group mean of the J-th group of category k
      for (j in seq_len(jlist)) {
        sum <- sum(y[jlist[[j]], 1])
        alpha[j, 1] <- sum / length(jlist[[j]])
      }

      # Update \alpha_{k}
      Alpha[[k]] <- alpha
    }

    # Compute termination criterion and check convergence
    crit <- sqrt(sum((unlist(Alpha) - unlist(Alpha0))^2) / sum(unlist(Alpha0)^2))

    if (crit < tol) {
      break
    }
  }

  return(Alpha)
}
