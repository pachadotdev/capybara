#' @title Split formula on top-level pipes
#' @description Splits formula string on `|` while respecting parentheses.
#'   Only splits on `|` that appear at parenthesis depth 0.
#' @param fml_chr Formula as character string
#' @return Character vector of formula parts
#' @noRd
split_formula_on_pipe_ <- function(fml_chr) {
  parts <- character(0L)
  current_part <- ""
  paren_depth <- 0L

  for (i in seq_len(nchar(fml_chr))) {
    char <- substr(fml_chr, i, i)

    if (char == "(") {
      paren_depth <- paren_depth + 1L
      current_part <- paste0(current_part, char)
    } else if (char == ")") {
      paren_depth <- paren_depth - 1L
      current_part <- paste0(current_part, char)
    } else if (char == "|" && paren_depth == 0L) {
      parts <- c(parts, trimws(current_part))
      current_part <- ""
    } else {
      current_part <- paste0(current_part, char)
    }
  }

  # Add the last part
  if (nzchar(current_part)) {
    parts <- c(parts, trimws(current_part))
  }

  parts
}

#' @title Remove unmatched parentheses from a formula part
#' @description Removes orphaned parentheses that result from splitting a
#'   wrapped formula. Handles cases like "(x" -> "x" and "y)" -> "y".
#' @param part Formula part string
#' @noRd
clean_formula_part_ <- function(part) {
  if (!nzchar(part)) {
    return(part)
  }

  part <- trimws(part)

  # Remove unmatched opening parens at the end (e.g., "x + y (" -> "x + y")
  while (nchar(part) > 0L && substr(part, nchar(part), nchar(part)) == "(") {
    part <- trimws(substr(part, 1L, nchar(part) - 1L))
  }

  # Remove unmatched closing parens at the start (e.g., ") + x" -> "+ x")
  while (nchar(part) > 0L && substr(part, 1L, 1L) == ")") {
    part <- trimws(substr(part, 2L, nchar(part)))
  }

  # Remove matching outer parentheses (e.g., "(x + y)" -> "x + y")
  while (nchar(part) >= 2L &&
    substr(part, 1L, 1L) == "(" &&
    substr(part, nchar(part), nchar(part)) == ")") {
    # Check if outer parens are actually matched
    paren_depth <- 0L
    matched <- TRUE
    for (i in seq_len(nchar(part))) {
      char <- substr(part, i, i)
      if (char == "(") {
        paren_depth <- paren_depth + 1L
      } else if (char == ")") {
        paren_depth <- paren_depth - 1L
        # If depth reaches 0 before the end, outer parens aren't matched
        if (paren_depth == 0L && i < nchar(part)) {
          matched <- FALSE
          break
        }
      }
    }
    # Only remove if outer parens are matched
    if (matched && paren_depth == 0L) {
      part <- substr(part, 2L, nchar(part) - 1L)
      part <- trimws(part)
    } else {
      break
    }
  }

  part
}


#' @title Split a |-separated felm / feglm formula into its string parts
#' @description
#' Returns a list with elements `base` (`"y ~ x"`), `fe` (or `NULL`), and
#' `cluster` (or `NULL`).  Used internally by [update.felm()] and
#' [update.feglm()] so that each segment can be updated independently before
#' reassembly.
#' @noRd
felm_formula_parts_ <- function(formula) {
  fml_chr <- deparse1(formula)
  parts <- split_formula_on_pipe_(fml_chr)

  # Clean unmatched parentheses from update.Formula() wrapping
  base <- clean_formula_part_(parts[[1L]])
  fe <- if (length(parts) >= 2L) clean_formula_part_(parts[[2L]]) else NULL
  cluster <- if (length(parts) >= 3L) clean_formula_part_(parts[[3L]]) else NULL

  list(
    base    = base,
    fe      = fe,
    cluster = cluster
  )
}

#' @noRd
felm_formula_ <- function(base, fe = NULL, cluster = NULL) {
  base_chr <- if (inherits(base, "formula")) {
    deparse1(base)
  } else {
    as.character(base)
  }

  parts <- base_chr

  if (!is.null(fe) && nzchar(trimws(fe))) {
    parts <- paste(parts, "|", fe)
  }

  if (!is.null(cluster) && nzchar(trimws(cluster))) {
    if (is.null(fe) || !nzchar(trimws(fe))) {
      stop(
        "`cluster` requires `fe` to be specified as well.\n",
        "Supply a fixed-effect variable via the `fe` argument.",
        call. = FALSE
      )
    }
    parts <- paste(parts, "|", cluster)
  }

  as.Formula(parts)
}

#' @title Update a fitted \code{felm} model
#' @description
#' S3 method for [update()] that understands the `|`-separated formula syntax
#' used by [felm()].  R's built-in [stats::update.formula()] breaks on these
#' formulas because the `|` parts look like factor arithmetic.  This method uses
#' the [Formula::update.Formula()] method which correctly handles multi-part
#' formulas.
#'
#' The `.` placeholder works as usual:
#' * `. ~ .` - keep the current response and RHS regressors.
#' * The second `|` segment replaces (or keeps, if `.`) the fixed-effects.
#' * The third `|` segment replaces (or keeps, if `.`) the cluster variables.
#'
#' @param object A fitted `felm` object.
#' @param formula. Update formula; only the segments you want to change need to
#'   differ from `.`.  Examples:
#'   * `. ~ . | country + year` - change FE, keep regressors.
#'   * `. ~ . | . | ctry1 + ctry2` - keep FE, change cluster.
#'   * `. ~ . - bothin | year` - drop a regressor, keep FE.
#' @param vcov Optional new `vcov` value (e.g. `"cluster"`).  If omitted the
#'   original value is reused.
#' @param ... Additional arguments forwarded to [felm()].
#'
#' @return A refitted `felm` object.
#' @export
update.felm <- function(object, formula. = . ~ ., vcov = NULL, ...) {
  # Convert to Formula object for proper multi-part formula handling
  old_fml <- Formula::as.Formula(object[["formula"]])
  new_fml <- update(old_fml, formula.)

  felm(
    formula = new_fml,
    data    = object[["data"]],
    control = object[["control"]],
    vcov    = if (!is.null(vcov)) vcov else object[["vcov_type"]],
    ...
  )
}

#' @title Update a fitted \code{feglm} model
#' @description
#' S3 method for [update()] that understands the `|`-separated formula syntax
#' used by [feglm()].  Uses [Formula::update.Formula()] for proper handling of
#' multi-part formulas.  Identical semantics to [update.felm()].
#'
#' @inheritParams update.felm
#' @param object A fitted `feglm` object.
#' @param family Optional new family (e.g. `binomial()`).  If omitted the
#'   original family is reused.
#' @return A refitted `feglm` object.
#' @export
update.feglm <- function(object, formula. = . ~ ., vcov = NULL, family = NULL, ...) {
  # Convert to Formula object for proper multi-part formula handling
  old_fml <- Formula::as.Formula(object[["formula"]])
  new_fml <- update(old_fml, formula.)

  feglm(
    formula = new_fml,
    data    = object[["data"]],
    family  = if (!is.null(family)) family else object[["family"]],
    control = object[["control"]],
    ...
  )
}

#' @title Update a \code{felm_formula} object
#' @description
#' S3 method for [update()] on `felm_formula` objects.  Uses
#' [Formula::update.Formula()] to properly handle multi-part formulas with `|`.
#'
#' The `.` placeholder behaves as in [stats::update.formula()]:
#' * `. ~ .` - keep current response and RHS unchanged.
#' * Second `|` segment - replaces (or keeps with `.`) the fixed effects.
#' * Third `|` segment - replaces (or keeps with `.`) the cluster variables.
#'
#' @param object A `felm_formula` object.
#' @param formula. Update specification, e.g. `. ~ . | . | ctry1 + ctry2`.
#' @param ... Ignored.
#'
#' @return A new `felm_formula` object.
#' @export
update.felm_formula <- function(object, formula., ...) {
  # Convert to Formula object, update, then restore felm_formula class
  fml <- Formula::as.Formula(object)
  updated_fml <- update(fml, formula.)

  # Restore the felm_formula class
  environment(updated_fml) <- environment(object)
  structure(updated_fml, class = c("felm_formula", class(updated_fml)))
}
