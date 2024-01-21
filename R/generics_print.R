summary_formula_ <- function(x) {
  cat("Formula: ")
  print(x[["formula"]])
}

summary_family_ <- function(x) {
  cat(
    "\nFamily: ", gsub("^([a-z])", "\\U\\1", x[["family"]][["family"]],
      perl = TRUE
    ), "\n",
    sep = ""
  )
}

summary_estimates_ <- function(x, digits) {
  cat("\nEstimates:\n\n")
  coefmat <- as.data.frame(x[["cm"]])

  coefmat[, max(ncol(coefmat))] <- sapply(coefmat[, max(nrow(coefmat))], function(x) {
    if (x <= 0.001) {
      paste(formatC(x, format = "f", digits = digits), "***")
    } else if (x <= 0.01) {
      paste(formatC(x, format = "f", digits = digits), "** ")
    } else if (x <= 0.05) {
      paste(formatC(x, format = "f", digits = digits), "*  ")
    } else if (x <= 0.1) {
      paste(formatC(x, format = "f", digits = digits), ".  ")
    } else {
      formatC(x, format = "f", digits = digits)
    }
  })

  # get rid of extra spaces (i.e., no number with ***)
  coefmat[, max(ncol(coefmat))] <- gsub("\\*\\s+$", "*", coefmat[, max(ncol(coefmat))])
  coefmat[, max(ncol(coefmat))] <- gsub("\\.\\s+$", ".", coefmat[, max(ncol(coefmat))])

  # fill coefmat[, max(ncol(coefmat))] with spaces to the right
  signif_width <- max(nchar(coefmat[, max(ncol(coefmat))]))
  coefmat[, max(ncol(coefmat))] <- sprintf("%-*s", signif_width, coefmat[, max(ncol(coefmat))])

  # format the other columns as formatC(x, format = "f", digits = digits)
  for (i in 1:(ncol(coefmat) - 1)) {
    coefmat[, i] <- formatC(as.double(coefmat[, i]), format = "f", digits = digits)
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
  cat("\nSignificance codes: *** 99.9%; ** 99%; * 95%; . 90%\n")
}

summary_r2_ <- function(x, digits) {
  cat(
    sprintf("\nR-squared%*s:", nchar("Adj. "), " "),
    format(x[["r.squared"]], digits = digits, nsmall = 2L), "\n"
  )
  cat(
    "Adj. R-squared:",
    format(x[["adj.r.squared"]], digits = digits, nsmall = 2L), "\n"
  )
}

summary_pseudo_rsq_ <- function(x, digits) {
  if (x[["family"]][["family"]] == "poisson") {
    cat(
      "\nPseudo R-squared:",
      format(x[["pseudo.rsq"]], digits = digits, nsmall = 2L), "\n"
    )
  }
}

summary_nobs_ <- function(x) {
  cat(
    "\nNumber of observations:",
    paste0("Full ", x[["nobs"]][["nobs"]], ";"),
    paste0("Missing ", x[["nobs"]][["nobs.na"]], ";"),
    paste0("Perfect classification ", x[["nobs"]][["nobs.pc"]]), "\n"
  )
}

summary_fisher_ <- function(x, digits) {
  if (is.null(x[["theta"]])) {
    cat("\nNumber of Fisher Scoring iterations:", x[["iter"]], "\n")
  } else {
    cat("\nNumber of Fisher Scoring iterations:", x[["iter"]])
    cat("\nNumber of outer iterations:", x[["iter.outer"]])
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
  summary_formula_(x)

  summary_family_(x)

  summary_estimates_(x, digits)

  summary_pseudo_rsq_(x, digits)

  summary_nobs_(x)

  summary_fisher_(x, digits)
}

#' @export
#' @noRd
print.summary.felm <- function(
    x, digits = max(3L, getOption("digits") - 3L), ...) {
  summary_formula_(x)

  summary_estimates_(x, digits)

  summary_r2_(x, digits)

  summary_nobs_(x)
}
