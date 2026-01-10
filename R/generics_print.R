#' srr_stats
#' @srrstats {G1.0} Implements `print` methods for various model objects (`apes`, `feglm`, `felm`) and their summaries.
#' @srrstats {G2.1a} Ensures that input objects are of the expected class (`apes`, `feglm`, `felm`, or summaries of
#'  these classes).
#' @srrstats {G3.2} Provides detailed output, including coefficients, significance levels, and iteration counts,
#'  tailored to the model type.
#' @srrstats {G3.3} Includes well-structured significance indicators (`***`, `**`, `*`, `.`) for coefficient p-values.
#' @srrstats {G5.2a} Outputs are formatted for clarity, with aligned columns and headers.
#' @srrstats {G5.4a} Validates consistency of printed summaries across model types, ensuring uniform presentation.
#' @srrstats {RE4.17} Specific default `print()` method for summaries and coefficients.
#' @srrstats {RE5.0} Reduces cyclomatic complexity by modularizing summary and print methods.
#' @srrstats {RE5.2} Facilitates easy interpretation of model summaries, including pseudo R-squared, deviance, and
#'  fixed-effects estimates.
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
  fam <- x[["family"]]
  cat(
    "\nFamily: ",
    gsub("^([a-z])", "\\U\\1", fam[["family"]], perl = TRUE),
    "\n",
    sep = ""
  )
}

#' @title Refactors for and 'feglm' summaries
#' @description Reduces the cyclomatic complexity of print.summary.feglm
#' @noRd
summary_estimates_ <- function(x, digits) {
  # Use pre-computed coefficient table from C++ (already has row/col names)
  coefficients <- x[["coef_table"]]

  # Skip printing if there are no slope coefficients (fixed effects only models)
  if (is.null(coefficients) || nrow(coefficients) == 0) {
    cat("\nNo slope coefficients\n")
    return(invisible(NULL))
  }

  cat("\nEstimates:\n\n")

  coefmat <- as.data.frame(coefficients)

  coefmat <- summary_estimates_signif_(coefmat, digits)
  coefmat <- summary_estimates_cols_(coefmat, digits)

  coef_width <- max(nchar(rownames(coefmat))) + 2L
  max_widths <- summary_estimates_max_width_(coefmat)

  summary_estimates_header_(coef_width, max_widths)
  summary_estimates_dashes_(coef_width, max_widths)
  summary_estimates_print_rows_(coefmat, coef_width, max_widths)

  cat("\nSignificance codes: ** p < 0.01; * p < 0.05; + p < 0.10\n")
}

summary_estimates_signif_ <- function(coefmat, digits) {
  last_col <- max(ncol(coefmat))
  pval_col <- coefmat[, last_col]

  pval_col <- vapply(
    pval_col,
    function(x) {
      formatted <- formatC(x, format = "f", digits = digits)
      if (x < 0.01) {
        paste(formatted, "** ")
      } else if (x < 0.05) {
        paste(formatted, "*  ")
      } else if (x < 0.1) {
        paste(formatted, "+  ")
      } else {
        formatted
      }
    },
    character(1)
  )

  pval_col <- gsub("\\*\\s+$", "*", pval_col)
  pval_col <- gsub("\\+\\s+$", "+", pval_col)

  signif_width <- max(nchar(pval_col))
  coefmat[, last_col] <- sprintf("%-*s", signif_width, pval_col)

  coefmat
}

summary_estimates_cols_ <- function(coefmat, digits) {
  n_cols <- ncol(coefmat) - 1
  coefmat[1:n_cols] <- lapply(coefmat[1:n_cols], function(col) {
    formatC(as.double(col), format = "f", digits = digits)
  })
  coefmat
}

summary_estimates_max_width_ <- function(coefmat) {
  # max_widths <- c(nchar("Estimate"), nchar("Std. Error"), nchar("t value"),
  #   nchar("Pr(>|t|)"))
  max_widths <- c(8L, 10L, 7L, 8L)

  Reduce(
    function(widths, i) {
      row_values <- coefmat[i, ]
      mapply(
        function(value, width) {
          max(width, nchar(value))
        },
        value = row_values,
        width = widths
      )
    },
    seq_len(nrow(coefmat)),
    init = max_widths
  )
}

summary_estimates_header_ <- function(coef_width, max_widths) {
  header <- mapply(
    function(name, width) {
      sprintf("| %-*s", width, name)
    },
    name = c("Estimate", "Std. Error", "z value", "Pr(>|z|)"),
    width = max_widths + 1L
  )

  cat(
    "|",
    paste(rep(" ", coef_width), collapse = ""),
    paste(header, collapse = ""),
    "|\n",
    sep = ""
  )
}

summary_estimates_dashes_ <- function(coef_width, max_widths) {
  dashes <- mapply(
    function(width) {
      sprintf("|%s", paste(rep("-", width), collapse = ""))
    },
    width = max_widths + 2L
  )

  cat(
    "|",
    paste(rep("-", coef_width), collapse = ""),
    paste(dashes, collapse = ""),
    "|\n",
    sep = ""
  )
}

summary_estimates_print_rows_ <- function(coefmat, coef_width, max_widths) {
  invisible(lapply(seq_len(nrow(coefmat)), function(i) {
    cat("| ", sprintf("%-*s", coef_width - 1L, rownames(coefmat)[i]), sep = "")
    row_values <- coefmat[i, ]
    formatted_values <- mapply(
      function(value, width) {
        sprintf("| %*s ", width, value)
      },
      value = row_values,
      width = max_widths
    )

    cat(formatted_values, "|\n", sep = "")
  }))
}

#' @title Refactors for and 'feglm' summaries
#' @description Reduces the cyclomatic complexity of print.summary.feglm
#' @noRd
summary_r2_ <- function(x, digits) {
  cat(
    sprintf("\nR-squared%*s:", nchar("Adj. "), " "),
    format(x[["r_squared"]], digits = digits, nsmall = 2L),
    "\n"
  )
  cat(
    "Adj. R-squared:",
    format(x[["adj_r_squared"]], digits = digits, nsmall = 2L),
    "\n"
  )
}

#' @title Refactors for and 'feglm' summaries
#' @description Reduces the cyclomatic complexity of print.summary.feglm
#' @noRd
summary_pseudo_rsq_ <- function(x, digits) {
  fam <- x[["family"]]
  if (fam[["family"]] == "poisson" && !is.null(x[["pseudo.rsq"]])) {
    cat(
      "\nPseudo R-squared:",
      format(x[["pseudo.rsq"]], digits = digits, nsmall = 2L),
      "\n"
    )
  }
}

#' @title Refactors for and 'feglm' summaries
#' @description Reduces the cyclomatic complexity of print.summary.feglm
#' @noRd
summary_nobs_ <- function(x) {
  nobs_vec <- x[["nobs"]]
  cat(
    "\nNumber of observations:",
    paste0("Full ", nobs_vec[["nobs"]], ";"),
    paste0("Missing ", nobs_vec[["nobs_na"]], ";"),
    paste0("Perfect classification ", nobs_vec[["nobs_pc"]]),
    "\n"
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

#' @title Print method for 'apes' objects (detailed output)
#' @description Similar to the 'print' method for 'glm' objects
#' @export
#' @noRd
print.apes <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  # Use pre-computed coefficient table from C++
  coefficients <- x[["vcov_table"]]
  if (is.null(coefficients)) {
    # Fallback for backward compatibility
    delta <- x[["delta"]]
    se <- sqrt(diag(x[["vcov"]]))
    coefficients <- cbind(
      Estimate = delta,
      `Std. Error` = se,
      `z value` = delta / se,
      `Pr(>|z|)` = 2.0 * pnorm(-abs(delta / se))
    )
    rownames(coefficients) <- names(delta)
  }

  # Skip printing if there are no slope coefficients
  if (!is.null(coefficients) && nrow(coefficients) > 0) {
    cat("Estimates:\n")
    printCoefmat(
      coefficients,
      P.values = TRUE,
      has.Pvalue = TRUE,
      digits = digits
    )
  } else {
    cat("No slope coefficients\n")
  }
  invisible(x)
}

#' @title Print method for 'feglm' objects (detailed output)
#' @description Similar to the 'print' method for 'glm' objects
#' @export
#' @noRd
print.feglm <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  summary_formula_(x)

  summary_family_(x)

  summary_estimates_(x, digits)

  summary_pseudo_rsq_(x, digits)

  summary_nobs_(x)

  summary_fisher_(x, digits)

  invisible(x)
}

#' @title Print method for 'summary.feglm' objects
#' @description Print method for feglm summary objects
#' @export
#' @noRd
print.summary.feglm <- function(
  x,
  digits = max(3L, getOption("digits") - 3L),
  ...
) {
  print.feglm(x, digits = digits, ...)
}

#' @title Print method for 'felm' objects (detailed output)
#' @description Similar to the 'print' method for 'lm' objects
#' @export
#' @noRd
print.felm <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  summary_formula_(x)

  summary_estimates_(x, digits)

  summary_r2_(x, digits)

  summary_nobs_(x)

  invisible(x)
}

#' @title Print method for 'summary.felm' objects
#' @description Print method for felm summary objects
#' @export
#' @noRd
print.summary.felm <- function(
  x,
  digits = max(3L, getOption("digits") - 3L),
  ...
) {
  print.felm(x, digits = digits, ...)
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

#' Print method for regression tables
#' @export
#' @noRd
print.capybara_separation <- function(x, ...) {
  cat("Separation Check Result\n")
  cat("-----------------------\n")
  cat("Separated observations:", x$num_separated, "\n")
  cat("Converged:", x$converged, "\n")
  if (x$num_separated > 0 && length(x$separated_obs) <= 20) {
    cat("Observation indices:", paste(x$separated_obs, collapse = ", "), "\n")
  } else if (x$num_separated > 0) {
    cat(
      "First 20 observation indices:",
      paste(head(x$separated_obs, 20), collapse = ", "),
      "...\n"
    )
  }
  invisible(x)
}
