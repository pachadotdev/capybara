#' srr_stats
#' @srrstats {G1.0} Implements `print` methods for various model objects (`apes`, `feglm`, `felm`) and their summaries.
#' @srrstats {G2.1a} Ensures that input objects are of the expected class (`apes`, `feglm`, `felm`, or summaries of these classes).
#' @srrstats {G3.2} Provides detailed output, including coefficients, significance levels, and iteration counts, tailored to the model type.
#' @srrstats {G3.3} Includes well-structured significance indicators (`***`, `**`, `*`, `.`) for coefficient p-values.
#' @srrstats {G5.2a} Outputs are formatted for clarity, with aligned columns and headers.
#' @srrstats {G5.4a} Validates consistency of printed summaries across model types, ensuring uniform presentation.
#' @srrstats {RE4.17} Specific default `print()` method for summaries and coefficients.
#' @srrstats {RE5.0} Reduces cyclomatic complexity by modularizing summary and print methods.
#' @srrstats {RE5.2} Facilitates easy interpretation of model summaries, including pseudo R-squared, deviance, and fixed-effects estimates.
#' @srrstats {RE5.3} Designed for extensibility to accommodate additional model types or summary elements.
#' @noRd
NULL

#' @title Refactors for and 'feglm' summaries
#' @description Reduces the cyclomatic complexity of print.summary.feglm
#' @noRd
summary_formula_ <- function(x) {
  cat("Formula: ")
  print(x[["formula"]])
}

#' @title Refactors for and 'feglm' summaries
#' @description Reduces the cyclomatic complexity of print.summary.feglm
#' @noRd
summary_family_ <- function(x) {
  cat(
    "\nFamily: ", gsub("^([a-z])", "\\U\\1", x[["family"]][["family"]],
      perl = TRUE
    ), "\n",
    sep = ""
  )
}

#' @title Refactors for and 'feglm' summaries
#' @description Reduces the cyclomatic complexity of print.summary.feglm
#' @noRd
summary_estimates_ <- function(x, digits) {
  cat("\nEstimates:\n\n")
  coefmat <- as.data.frame(x[["coefficients"]])

  coefmat <- summary_estimates_signif_(coefmat, digits)
  coefmat <- summary_estimates_cols_(coefmat, digits)

  coef_width <- max(nchar(rownames(coefmat))) + 2L
  max_widths <- summary_estimates_max_width_(coefmat)

  summary_estimates_header_(coef_width, max_widths)
  summary_estimates_dashes_(coef_width, max_widths)
  summary_estimates_print_rows_(coefmat, coef_width, max_widths)

  cat("\nSignificance codes: *** 99.9%; ** 99%; * 95%; . 90%\n")
}

summary_estimates_signif_ <- function(coefmat, digits) {
  coefmat[, max(ncol(coefmat))] <- vapply(
    coefmat[, max(ncol(coefmat))],
    function(x) {
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
    }, character(1)
  )

  coefmat[, max(ncol(coefmat))] <- gsub(
    "\\*\\s+$", "*",
    coefmat[, max(ncol(coefmat))]
  )
  coefmat[, max(ncol(coefmat))] <- gsub(
    "\\.\\s+$", ".",
    coefmat[, max(ncol(coefmat))]
  )

  signif_width <- max(nchar(coefmat[, max(ncol(coefmat))]))
  coefmat[, max(ncol(coefmat))] <- sprintf(
    "%-*s", signif_width,
    coefmat[, max(ncol(coefmat))]
  )

  coefmat
}

summary_estimates_cols_ <- function(coefmat, digits) {
  for (i in 1:(ncol(coefmat) - 1)) {
    coefmat[, i] <- formatC(as.double(coefmat[, i]),
      format = "f",
      digits = digits
    )
  }
  coefmat
}

summary_estimates_max_width_ <- function(coefmat) {
  # max_widths <- c(nchar("Estimate"), nchar("Std. Error"), nchar("t value"),
  #   nchar("Pr(>|t|)"))
  max_widths <- c(8L, 10L, 7L, 8L)

  for (i in seq_len(nrow(coefmat))) {
    row_values <- coefmat[i, ]
    max_widths <- mapply(function(value, width) {
      max(width, nchar(value))
    }, value = row_values, width = max_widths)
  }

  max_widths
}

summary_estimates_header_ <- function(coef_width, max_widths) {
  header <- mapply(
    function(name, width) {
      sprintf("| %-*s", width, name)
    },
    name = c("Estimate", "Std. Error", "z value", "Pr(>|z|)"),
    width = max_widths + 1L
  )

  cat("|", paste(rep(" ", coef_width), collapse = ""),
    paste(header, collapse = ""), "|\n",
    sep = ""
  )
}

summary_estimates_dashes_ <- function(coef_width, max_widths) {
  dashes <- mapply(function(width) {
    sprintf("|%s", paste(rep("-", width), collapse = ""))
  }, width = max_widths + 2L)

  cat("|", paste(rep("-", coef_width), collapse = ""),
    paste(dashes, collapse = ""), "|\n",
    sep = ""
  )
}

summary_estimates_print_rows_ <- function(coefmat, coef_width, max_widths) {
  for (i in seq_len(nrow(coefmat))) {
    cat("| ", sprintf("%-*s", coef_width - 1L, rownames(coefmat)[i]), sep = "")
    row_values <- coefmat[i, ]
    formatted_values <- mapply(function(value, width) {
      sprintf("| %*s ", width, value)
    }, value = row_values, width = max_widths)

    cat(formatted_values, "|\n", sep = "")
  }
}

#' @title Refactors for and 'feglm' summaries
#' @description Reduces the cyclomatic complexity of print.summary.feglm
#' @noRd
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

#' @title Refactors for and 'feglm' summaries
#' @description Reduces the cyclomatic complexity of print.summary.feglm
#' @noRd
summary_pseudo_rsq_ <- function(x, digits) {
  if (x[["family"]][["family"]] == "poisson") {
    cat(
      "\nPseudo R-squared:",
      format(x[["pseudo.rsq"]], digits = digits, nsmall = 2L), "\n"
    )
  }
}

#' @title Refactors for and 'feglm' summaries
#' @description Reduces the cyclomatic complexity of print.summary.feglm
#' @noRd
summary_nobs_ <- function(x) {
  cat(
    "\nNumber of observations:",
    paste0("Full ", x[["nobs"]][["nobs"]], ";"),
    paste0("Missing ", x[["nobs"]][["nobs_na"]], ";"),
    paste0("Perfect classification ", x[["nobs"]][["nobs_pc"]]), "\n"
  )
}

#' @title Refactors for and 'feglm' summaries
#' @description Reduces the cyclomatic complexity of print.summary.feglm
#' @noRd
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

#' @title Print method for 'apes' objects
#' @description Similar to the 'print' method for 'glm' objects
#' @export
#' @noRd
print.apes <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  print(x[["delta"]], digits = digits)
}

#' @title Print method for 'feglm' objects
#' @description Similar to the 'print' method for 'glm' objects
#' @export
#' @noRd
print.feglm <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(
    sub("\\(.*\\)", "", x[["family"]][["family"]]), " - ",
    x[["family"]][["link"]], " link",
    ", l= [", paste0(x[["lvls_k"]], collapse = ", "), "]\n\n",
    sep = ""
  )
  print(x[["coefficients"]], digits = digits)
}

#' @title Print method for 'felm' objects
#' @description Similar to the 'print' method for 'lm' objects
#' @export
#' @noRd
print.felm <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  print(x[["coefficients"]], digits = digits)
}

#' @title Print method for 'apes' summary objects
#' @description Similar to the 'print' method for 'glm' objects
#' @export
#' @noRd
print.summary.apes <- function(
    x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat("Estimates:\n")
  printCoefmat(x[["coefficients"]], P.values = TRUE, has.Pvalue = TRUE, digits = digits)
}

#' @title Print method for 'feglm' summary objects
#' @description Similar to the 'print' method for 'glm' objects
#' @export
#' @noRd
print.summary.feglm <- function(
    x, digits = max(3L, getOption("digits") - 3L),
    ...) {
  summary_formula_(x)

  summary_family_(x)

  summary_estimates_(x, digits)

  summary_pseudo_rsq_(x, digits)

  summary_nobs_(x)

  summary_fisher_(x, digits)
}

#' @title Print method for 'felm' summary objects
#' @description Similar to the 'print' method for 'lm' objects
#' @export
#' @noRd
print.summary.felm <- function(
    x, digits = max(3L, getOption("digits") - 3L), ...) {
  summary_formula_(x)

  summary_estimates_(x, digits)

  summary_r2_(x, digits)

  summary_nobs_(x)
}

#' Print method for regression tables
#' @param x A summary_table object
#' @param ... Additional arguments passed to other methods
#' @export
#' @noRd
print.summary_table <- function(x, ...) {
  cat(x$content, sep = "\n")
  invisible(x)
}
