#' @export
#' @noRd
print.apes <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  print(x[["delta"]], digits = digits)
}

#' @export
#' @noRd
print.feglm <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(
    sub("\\(.*\\)", "", x[["family"]][["family"]]), " - ",
    x[["family"]][["link"]], " link",
    ", l= [", paste0(x[["lvls.k"]], collapse = ", "), "]\n\n",
    sep = ""
  )
  print(x[["coefficients"]], digits = digits)
}

#' @export
#' @noRd
print.felm <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  print(x[["coefficients"]], digits = digits)
}

#' @export
#' @noRd
print.summary.apes <- function(
    x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat("Estimates:\n")
  printCoefmat(x[["cm"]], P.values = TRUE, has.Pvalue = TRUE, digits = digits)
}

#' @export
#' @noRd
print.summary.feglm <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(
    sub("\\(.*\\)", "", x[["family"]][["family"]]), " - ",
    x[["family"]][["link"]], " link\n\n",
    sep = ""
  )
  print(x[["formula"]])
  cat("\nEstimates:\n")
  printCoefmat(x[["cm"]], P.values = TRUE, has.Pvalue = TRUE, digits = digits)
  cat(
    "\nresidual deviance= ",
    format(x[["deviance"]], digits = max(5L, digits + 1L), nsmall = 2L),
    ",\n",
    sep = ""
  )
  cat(
    "null deviance= ",
    format(x[["null.deviance"]], digits = max(5L, digits + 1L), nsmall = 2L),
    ",\n",
    sep = ""
  )
  cat(
    "n= ", x[["nobs"]][["nobs"]],
    ", l= [", paste0(x[["lvls.k"]], collapse = ", "), "]\n",
    sep = ""
  )
  if (x[["nobs"]][["nobs.na"]] > 0L | x[["nobs"]][["nobs.pc"]] > 0L) {
    cat("\n")
    if (x[["nobs"]][["nobs.na"]] > 0L) {
      cat("(", x[["nobs"]][["nobs.na"]], "observation(s) deleted due to missingness )\n")
    }
    if (x[["nobs"]][["nobs.pc"]] > 0L) {
      cat("(", x[["nobs"]][["nobs.pc"]], "observation(s) deleted due to perfect classification )\n")
    }
  }
  if (is.null(x[["theta"]])) {
    cat("\nNumber of Fisher Scoring Iterations:", x[["iter"]], "\n")
  } else {
    cat("\nNumber of Fisher Scoring Iterations:", x[["iter"]])
    cat("\nNumber of Outer Iterations:", x[["iter.outer"]])
    cat(
      "\ntheta= ",
      format(x[["theta"]], digits = digits, nsmall = 2L),
      ", std. error= ",
      format(attr(x[["theta"]], "SE"), digits = digits, nsmall = 2L),
      "\n",
      sep = ""
    )
  }
}

#' @export
#' @noRd
print.summary.felm <- function(
    x, digits = max(3L, getOption("digits") - 3L), ...) {
  print(x[["formula"]])
  cat("\nEstimates:\n")
  printCoefmat(x[["cm"]], P.values = TRUE, has.Pvalue = TRUE, digits = digits)

  cat("\nR-square: ", format(x[["r.squared"]], digits = digits, nsmall = 2L), "\n")
  cat(
    "Adj. R-square: ",
    format(x[["adj.r.squared"]], digits = digits, nsmall = 2L),
    "\n"
  )

  # f <- x$fitted.values
  # w <- x$weights

  # x$cm

  # if (p != attr(x$terms, "intercept")) {
  #   df.int <- if (attr(z$terms, "intercept")) 1L else 0L
  #   ans$r.squared <- mss / (mss + rss)
  #   ans$adj.r.squared <- 1 - (1 - ans$r.squared) * ((n - df.int) / rdf)
  #   ans$fstatistic <- c(
  #     value = (mss / (p - df.int)) / resvar,
  #     numdf = p - df.int, dendf = rdf
  #   )
  # } else {
  #   ans$r.squared <- ans$adj.r.squared <- 0
  # }
}
