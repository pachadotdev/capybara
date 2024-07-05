#' @title Kendall Correlation
#' 
#' @description \code{\link{kendall_cor}} calculates the Kendall correlation
#'  coefficient between two numeric vectors. It uses the algorithm described in
#'  Knight (1966), which is based on the number of concordant and discordant
#'  pairs. The computational complexity of the algorithm is
#'  \eqn{O(n \log(n))}{O(n log(n))}, which is faster than the base R
#'  implementation in \code{\link[stats]{cor}} with \code{method = "kendall"}
#'  that has a computational complexity of \eqn{O(n^2)}{O(n^2)}. For small
#'  vectors (i.e., less than 100 observations), the time difference is
#'  negligible. However, for larger vectors, the difference can be substantial.
#' 
#'  By construction, the implementation drops missing values on a pairwise
#'  basis. This is the same as using \code{\link[stats]{cor}} with
#'  \code{use = "pairwise.complete.obs"}.
#' 
#' @param x a numeric vector.
#' @param y a numeric vector.
#' 
#' @return A numeric value between -1 and 1.
#' 
#' @references Knight, W. R. (1966). "A Computer Method for Calculating
#'  Kendall's Tau with Ungrouped Data". Journal of the American Statistical
#'  Association, 61(314), 436–439.
#' 
#' @examples
#' x <- c(1, 0, 2)
#' y <- c(5, 3, 4)
#' kendall_cor(x, y)
#' 
#' @export
kendall_cor <- function(x, y) {
  arr <- cbind(x, y)
  storage.mode(arr) <- "double"
  arr <- arr[complete.cases(arr), ]

  kw <- kendall_warnings(arr)

  if (isFALSE(kw)) {
    return(NA_real_)
  }

  kendall_cor_(arr)
}

#' @title Kendall Correlation Test
#' 
#' @description \code{\link{kendall_cor_test}} calculates p-value for the the
#'  Kendall correlation using the exact values when the number of observations
#'  is less than 50. For larger samples, it uses an approximation as in base R.
#'
#' @param x a numeric vector.
#' @param y a numeric vector.
#' @param alternative a character string specifying the alternative hypothesis.
#'  The possible values are \code{"two.sided"}, \code{"greater"}, and
#'  \code{"less"}.
#' 
#' @return A list with the following components:
#' \item{statistic}{The Kendall correlation coefficient.}
#' \item{p_value}{The p-value of the test.}
#' \item{alternative}{A character string describing the alternative hypothesis.}
#' 
#' @references Knight, W. R. (1966). "A Computer Method for Calculating
#'  Kendall's Tau with Ungrouped Data". Journal of the American Statistical
#'  Association, 61(314), 436–439.
#' 
#' @examples
#' x <- c(1, 0, 2)
#' y <- c(5, 3, 4)
#' kendall_cor_test(x, y)
#' 
#' @export
kendall_cor_test <- function(x,y,
  alternative = c("two.sided", "greater", "less")) {
  alternative <- match.arg(alternative)

  arr <- cbind(x, y)
  storage.mode(arr) <- "double"
  arr <- arr[complete.cases(arr), ]

  kw <- kendall_warnings(arr)

  if (isFALSE(kw)) {
    return(NA)
  }

  r <- kendall_cor_(arr)
  n <- nrow(arr)

  if (n < 2) {
    stop("not enough finite observations")
  }
  
  # r = correlation
  # n = number of observations

  if (n < 50) {
    q <- round((r + 1) * n * (n - 1) / 4)
    pv <- switch(alternative,
      "two.sided" = {
        if (q > n * (n - 1) / 4) {
          p <- 1 - pkendall_(q - 1, n)
        } else {
          p <- pkendall_(q, n)
        }
          min(2 * p, 1)
        },
      "greater" = 1 - pkendall_(q - 1, n),
      "less" = pkendall_(q, n)
    )
  } else {
    xties <- table(x[duplicated(x)]) + 1
    yties <- table(y[duplicated(y)]) + 1
    T0 <- n * (n - 1) / 2
    T1 <- sum(xties * (xties - 1)) / 2
    T2 <- sum(yties * (yties - 1)) / 2
    v0 <- n * (n - 1) * (2 * n + 5)
    vt <- sum(xties * (xties - 1) * (2 * xties + 5))
    vu <- sum(yties * (yties - 1) * (2 * yties + 5))
    v1 <- sum(xties * (xties - 1)) * sum(yties * (yties - 1))
    v2 <- sum(xties * (xties - 1) * (xties - 2)) *
      sum(yties * (yties - 1) * (yties - 2))
    var_S <- (v0 - vt - vu) / 18 +
      v1 / (2 * n * (n - 1)) +
      v2 / (9 * n * (n - 1) * (n - 2))
    S <- r * sqrt((T0 - T1) * (T0 - T2)) / sqrt(var_S)
    pv <- switch(alternative,
      "two.sided" = 2 * min(pnorm(S), pnorm(S, lower.tail = FALSE)),
      "greater" = pnorm(S, lower.tail = FALSE),
      "less" = pnorm(S)
    )
  }

  alt <- switch(
    alternative,
    "two.sided" = "alternative hypothesis: true tau is not equal to 0",
    "greater" = "alternative hypothesis: true tau is greater than 0",
    "less" = "alternative hypothesis: true tau is less than 0"
  )

  list(
    statistic = r,
    p_value = pv,
    alternative = alt
  )
}

kendall_warnings <- function(arr) {
  if (ncol(arr) != 2) {
    stop("x and y must be uni-dimensional vectors")
  }

  if (nrow(arr) < 2) {
    stop("x and y must have at least 2 observations")
  }

  if (sd(arr[, 1]) == 0) {
    warning("x has zero variance")
    return(FALSE)
  }

  if (sd(arr[, 2]) == 0) {
    warning("y has zero variance")
    return(FALSE)
  }

  TRUE
}
