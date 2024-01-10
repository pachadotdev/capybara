#' Get alpha for each category k
#' @param pie vector
#' @param klist list
#' @param tol numeric
#' @param maxiter integer
get_alpha_r_ <- function(pie, klist, tol, maxiter = 10000L) {
  n <- length(pie)
  K <- length(klist)

  Alpha <- lapply(klist, function(jlist) rep(0, length(jlist)))

  for (iter in seq_len(maxiter)) {
    Alpha0 <- Alpha

    for (k in 1:K) {
      y <- pie
      for (kk in 1:K) {
        if (kk != k) {
          jlist <- klist[[kk]]
          J <- length(jlist)
          for (j in 1:J) {
            indexes <- jlist[[j]]
            
            # for (i in indexes) {
            #   y[i] <- y[i] - Alpha[[kk]][j]
            # }
            
            y[indexes] <- y[indexes] - Alpha[[kk]][j]
          }
        }
      }

      jlist <- klist[[k]]
      J <- length(jlist)
      alpha <- sapply(jlist, function(indexes) {
        sum(y[indexes + 1]) / length(indexes)
      })

      Alpha[[k]] <- alpha
    }

    num <- sum(sapply(1:K, function(k) sum((Alpha[[k]] - Alpha0[[k]])^2)))
    denom <- sum(sapply(1:K, function(k) sum(Alpha0[[k]]^2)))
    crit <- sqrt(num / denom)
    if (crit < tol) {
      break
    }
  }

  if (iter == maxiter) {
    warning("Maximum number of iterations reached without convergence")
  }

  # convert vectors to column vectors
  Alpha <- lapply(Alpha, function(x) matrix(x, nrow = length(x), ncol = 1))

  return(Alpha)
}
