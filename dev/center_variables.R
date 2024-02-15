#' @keywords internal
#' @noRd
center_variables_ <- function(V, w, klist, tol, maxiter) {
  # Auxiliary variables (fixed)
  N <- nrow(V)
  P <- ncol(V)
  K <- length(klist)

  sw <- sum(w)

  # Auxiliary variables (storage)
  C <- matrix(0, N, P)

  # Halperin projections
  for (p in seq_len(P)) {
    # Center each variable
    x <- V[, p]

    for (iter in seq_len(maxiter)) {
      # Store centered vector from the last iteration
      x0 <- x

      # Compute all weighted group means of category 'k' and subtract them
      for (k in seq_len(K)) {
        # Demeaned j-th group of category 'k'
        for (j in seq_along(klist[[k]])) {
          x[klist[[k]][[j]]] <- x[klist[[k]][[j]]] - weighted.mean(
            x[klist[[k]][[j]]],
            w[klist[[k]][[j]]]
          )
        }
      }

      # Check convergence
      delta <- sum(abs(x - x0) / (1 + abs(x0)) * w) / sw

      if (delta < tol) {
        break
      }
    }

    C[, p] <- x
  }

  C
}
