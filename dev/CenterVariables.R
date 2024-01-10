# Base version, without any optimization
# This works but the performance is not good enough
# See the more efficient implementation below

# centerVariables <- function(V, w, klist, tol, maxiter = 100000L) {
#   # Auxiliary variables (fixed)
#   nRows <- nrow(V)
#   numCategories <- length(klist)
#   nCols <- ncol(V)
#   sumWeights <- sum(w)

#   # Auxiliary variables (storage)
#   M <- matrix(0, nRows, nCols)

#   # Halperin projections
#   for (colIndex in seq_len(nCols)) {
#     # Center each variable
#     x <- V[, colIndex]

#     for (iteration in seq_len(maxiter)) {
#       # Store centered vector from the last iteration
#       xPrev <- x

#       # Alternate between categories
#       for (catIndex in seq_len(numCategories)) {
#         # Compute all weighted group means of category and subtract them
#         groupList <- klist[[catIndex]]
#         numGroups <- length(groupList)

#         for (groupIndex in 1:numGroups) {
#           # Subset group of category
#           indexes <- groupList[[groupIndex]]
#           if (length(indexes) == 0) next # Skip if the group is empty

#           # Adjust the index for R's 1-based indexing
#           adjustedIndexes <- indexes + 1

#           # Compute numerator and denominator of the weighted group mean
#           weights <- w[adjustedIndexes]
#           xSubset <- x[adjustedIndexes]
#           numerator <- sum(weights * xSubset)
#           denominator <- sum(weights)

#           # Subtract weighted group mean
#           # meanGroup <- numerator / denominator

#           # Avoid division by zero
#           meanGroup <- ifelse(denominator == 0, Inf, numerator / denominator)

#           x[adjustedIndexes] <- xSubset - meanGroup
#         }
#       }

#       # Check convergence
#       delta <- sum(abs(x - xPrev) / (1 + abs(xPrev)) * w) / sumWeights
#       if (delta < tol) {
#         break
#       }
#     }

#     M[, colIndex] <- x
#   }

#   # Return matrix with centered variables
#   return(M)
# }

#' Center variables
#' @param V matrix
#' @param w vector
#' @param klist list
#' @param tol numeric
#' @param maxiter integer
centerVariables <- function(V, w, klist, tol, maxiter = 100000L) {
  nRows <- nrow(V)
  nCols <- ncol(V)
  sumWeights <- sum(w)
  M <- matrix(0, nRows, nCols)

  # Precompute adjusted indices and weight sums
  adjustedIndices <- lapply(klist, function(category) {
    lapply(category, function(g) {
      return(g + 1)
    })
  })
  weightSums <- lapply(adjustedIndices, function(cat) {
    vapply(cat, function(g) {
      return(sum(w[g]))
    }, FUN.VALUE = numeric(1))
  })

  for (colIndex in seq_len(nCols)) {
    x <- V[, colIndex]
    xPrev <- x
    for (iteration in seq_len(maxiter)) {
      # Category processing
      for (catIndex in seq_len(length(klist))) {
        category <- adjustedIndices[[catIndex]]
        sumsCat <- weightSums[[catIndex]]

        for (groupIndex in seq_along(category)) {
          indices <- category[[groupIndex]]
          if (length(indices) == 0) next
          groupWeights <- w[indices]
          xSubset <- x[indices]
          meanGroup <- sum(groupWeights * xSubset) / sumsCat[groupIndex]
          x[indices] <- xSubset - meanGroup
        }
      }

      # Check for convergence
      delta <- sum(abs(x - xPrev) / (1 + abs(xPrev)) * w) / sumWeights
      if (delta < tol) break
      xPrev <- x
    }
    M[, colIndex] <- x
  }

  return(M)
}
