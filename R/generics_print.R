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
  cat("Formula:\n")
  print(x[["formula"]])
  cat("\nEstimates:\n")
  coefmat <- as.data.frame(x[["cm"]])

  coefmat[, max(ncol(coefmat))] <- sapply(coefmat[, max(nrow(coefmat))], function(x) {
    # significance codes for Pr(>|t|):
    # 0.1%: (***)
    # 1%: (**)
    # 5%: (*)
    # 10%: (.)
    if (x <= 0.001) {
      paste(formatC(x, format = "f", digits = digits), "(***)")
    } else if (x <= 0.01) {
      paste(formatC(x, format = "f", digits = digits), "(**)")
    } else if (x <= 0.05) {
      paste(formatC(x, format = "f", digits = digits), "(*)")
    } else if (x <= 0.1) {
      paste(formatC(x, format = "f", digits = digits), "(.)")
    } else {
      formatC(x, format = "f", digits = digits)
    }
  })

  # format the other columns as formatC(x, format = "f", digits = digits)
  for (i in 1:(ncol(coefmat) - 1)) {
    coefmat[, i] <- sapply(coefmat[, i], function(x) {
      formatC(as.double(x), format = "f", digits = digits)
    })
  }

  coef_width <- max(nchar(rownames(coefmat))) + 2L
  max_widths <- c(nchar("Estimate"), nchar("Std. Error"), nchar("t value"), nchar("Pr(>|t|)"))

  # get the maximum number of digits (with sign and decimal point) for each column
  for (i in 1:nrow(coefmat)) {
    row_values <- coefmat[i, ]
    max_widths <- mapply(function(value, width) {
      max(width, nchar(value))
    }, value = row_values, width = max_widths)
  }

  # create a header such as "| Estimate | Std. Error | t value | Pr(>|t|) |\n"
  # but adding spaces between the bars and the column names to make sure the numbers will fit
  header <- mapply(function(name, width) {
    sprintf("| %-*s", width, name)
  }, name = colnames(coefmat), width = max_widths + 1L)

  cat("|", paste(rep(" ", coef_width), collapse = ""), paste(header, collapse = ""), "|\n", sep = "")

  # now the same for "|----|----|----|----|\n"
  dashes <- mapply(function(width) {
    sprintf("|%s", paste(rep("-", width), collapse = ""))
  }, width = max_widths + 2L)

  cat("|", paste(rep("-", coef_width), collapse = ""), paste(dashes, collapse = ""), "|\n", sep = "")

  for (i in 1:nrow(coefmat)) {
    cat("| ", sprintf("%-*s", coef_width - 1L, rownames(coefmat)[i]), sep = "")
    row_values <- coefmat[i, ]
    formatted_values <- mapply(function(value, width) {
      sprintf("| %*s ", width, value)
    }, value = row_values, width = max_widths)

    if (i == max(nrow(coefmat))) {
      cat(formatted_values, "|\n", sep = "")
    } else {
      cat(formatted_values, "|\n", sep = "")
    }
  }

  # significance message
  cat("\nSignificance codes:  (***) 0.1%; (**) 1%; (*) 5%; (.) 10%\n")

  cat(
    sprintf("\nR-squared%*s:", nchar("Adj. "), " "),
    format(x[["r.squared"]], digits = digits, nsmall = 2L), "\n"
  )
  cat(
    "Adj. R-squared:",
    format(x[["adj.r.squared"]], digits = digits, nsmall = 2L), "\n"
  )
}
