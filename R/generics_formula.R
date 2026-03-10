#' @title Split a |-separated felm / feglm formula into its string parts
#' @description
#' Returns a list with elements `base` (`"y ~ x"`), `fe` (or `NULL`), and
#' `cluster` (or `NULL`).  Used internally by [update.felm()] and
#' [update.feglm()] so that each segment can be updated independently before
#' reassembly via [felm_formula()].
#' @noRd
felm_formula_parts_ <- function(formula) {
  fml_chr <- deparse1(formula)
  parts <- trimws(strsplit(fml_chr, "\\|")[[1L]])
  list(
    base    = parts[[1L]],
    fe      = if (length(parts) >= 2L) parts[[2L]] else NULL,
    cluster = if (length(parts) >= 3L) parts[[3L]] else NULL
  )
}

#' @title Update a fitted \code{felm} model
#' @description
#' S3 method for [update()] that understands the `|`-separated formula syntax
#' used by [felm()].  R's built-in [update.formula()] breaks on these formulas
#' because the `|` parts look like factor arithmetic.  This method splits the
#' formula into its three segments, applies updates to each segment
#' independently, reassembles, and re-fits the model.
#'
#' The `.` placeholder works as usual:
#' * `. ~ .` — keep the current response and RHS regressors.
#' * The second `|` segment replaces (or keeps, if `.`) the fixed-effects.
#' * The third `|` segment replaces (or keeps, if `.`) the cluster variables.
#'
#' @param object A fitted `felm` object.
#' @param formula. Update formula; only the segments you want to change need to
#'   differ from `.`.  Examples:
#'   * `. ~ . | country + year` — change FE, keep regressors.
#'   * `. ~ . | . | ctry1 + ctry2` — keep FE, change cluster.
#'   * `. ~ . - bothin | year` — drop a regressor, keep FE.
#' @param vcov Optional new `vcov` value (e.g. `"cluster"`).  If omitted the
#'   original value is reused.
#' @param ... Additional arguments forwarded to [felm()].
#'
#' @return A refitted `felm` object.
#' @export
update.felm <- function(object, formula. = . ~ ., vcov = NULL, ...) {
  old <- felm_formula_parts_(object[["formula"]])

  new_chr  <- deparse1(formula.)
  new_segs <- trimws(strsplit(new_chr, "\\|")[[1L]])

  # --- base (lhs ~ rhs) ------------------------------------------------
  new_base <- deparse1(
    update.formula(as.formula(old$base), as.formula(new_segs[[1L]]))
  )

  # --- FE segment -------------------------------------------------------
  new_fe <- if (length(new_segs) >= 2L) {
    seg <- new_segs[[2L]]
    if (seg == ".") old$fe else if (nzchar(seg)) seg else NULL
  } else {
    old$fe
  }

  # --- cluster segment --------------------------------------------------
  new_cluster <- if (length(new_segs) >= 3L) {
    seg <- new_segs[[3L]]
    if (seg == ".") old$cluster else if (nzchar(seg)) seg else NULL
  } else {
    old$cluster
  }

  new_formula <- felm_formula(new_base, fe = new_fe, cluster = new_cluster)

  felm(
    formula = new_formula,
    data    = object[["data"]],
    control = object[["control"]],
    vcov    = if (!is.null(vcov)) vcov else object[["vcov_type"]],
    ...
  )
}

#' @title Update a fitted \code{feglm} model
#' @description
#' S3 method for [update()] that understands the `|`-separated formula syntax
#' used by [feglm()].  Identical semantics to [update.felm()].
#'
#' @inheritParams update.felm
#' @param object A fitted `feglm` object.
#' @param family Optional new family (e.g. `binomial()`).  If omitted the
#'   original family is reused.
#' @return A refitted `feglm` object.
#' @export
update.feglm <- function(object, formula. = . ~ ., vcov = NULL, family = NULL, ...) {
  old <- felm_formula_parts_(object[["formula"]])

  new_chr  <- deparse1(formula.)
  new_segs <- trimws(strsplit(new_chr, "\\|")[[1L]])

  new_base <- deparse1(
    update.formula(as.formula(old$base), as.formula(new_segs[[1L]]))
  )

  new_fe <- if (length(new_segs) >= 2L) {
    seg <- new_segs[[2L]]
    if (seg == ".") old$fe else if (nzchar(seg)) seg else NULL
  } else {
    old$fe
  }

  new_cluster <- if (length(new_segs) >= 3L) {
    seg <- new_segs[[3L]]
    if (seg == ".") old$cluster else if (nzchar(seg)) seg else NULL
  } else {
    old$cluster
  }

  new_formula <- felm_formula(new_base, fe = new_fe, cluster = new_cluster)

  feglm(
    formula = new_formula,
    data    = object[["data"]],
    family  = if (!is.null(family)) family else object[["family"]],
    control = object[["control"]],
    ...
  )
}

#' @title \code{|}-aware \code{update} for formulas
#' @description
#' Overrides [stats::update.formula()] so that formulas containing `|`
#' (i.e. the fixed-effect / cluster syntax used by [felm()] and [feglm()])
#' can be updated with the usual `.` placeholder syntax without triggering
#' *"'|' not meaningful for factors"* errors.
#'
#' Formulas that contain no `|` are forwarded unchanged to
#' [stats::update.formula()], so existing code is unaffected.
#'
#' @param object A `formula` object, possibly containing `|` separators.
#' @param formula. The update formula, e.g. `. ~ . | . | ctry1 + ctry2`.
#' @param ... Passed to [stats::update.formula()] for plain formulas.
#'
#' @return An updated `formula`.
#' @export
update.formula <- function(object, formula., ...) {
  obj_chr <- deparse1(object)
  new_chr <- deparse1(formula.)

  # Delegate to stats for plain formulas (no | in either side)
  if (!grepl("|", obj_chr, fixed = TRUE) &&
      !grepl("|", new_chr, fixed = TRUE)) {
    return(stats::update.formula(object, formula., ...))
  }

  old <- felm_formula_parts_(object)
  new_segs <- trimws(strsplit(new_chr, "\\|")[[1L]])

  # base (lhs ~ rhs) — safe to delegate to stats
  new_base <- deparse1(
    stats::update.formula(as.formula(old$base), as.formula(new_segs[[1L]]))
  )

  # FE segment
  new_fe <- if (length(new_segs) >= 2L) {
    seg <- new_segs[[2L]]
    if (seg == ".") old$fe else if (nzchar(seg)) seg else NULL
  } else {
    old$fe
  }

  # cluster segment
  new_cluster <- if (length(new_segs) >= 3L) {
    seg <- new_segs[[3L]]
    if (seg == ".") old$cluster else if (nzchar(seg)) seg else NULL
  } else {
    old$cluster
  }

  felm_formula(new_base, fe = new_fe, cluster = new_cluster)
}

#' @title Create a \code{|}-aware formula for felm / feglm
#' @description
#' Wraps a `|`-separated formula in the subclass `"felm_formula"` so that
#' [update()] dispatches to [update.felm_formula()] instead of base R's
#' [update.formula()], which cannot handle the `|` syntax and raises
#' *"'|' not meaningful for factors"* errors.
#'
#' The returned object is fully compatible with [felm()] and [feglm()].
#'
#' @param formula A formula of the form `y ~ x`, `y ~ x | fe`, or
#'   `y ~ x | fe | cluster`.
#'
#' @return The same formula with an additional `"felm_formula"` class prepended.
#'
#' @examples
#' fml <- felm_fml(ltrade ~ bothin + lrgdp | year)
#'
#' # update() now works safely on fml
#' update(fml, . ~ . | . | pair)
#' #> ltrade ~ bothin + lrgdp | year | pair
#'
#' @export
felm_fml <- function(formula) {
  structure(formula, class = c("felm_formula", "formula"))
}

#' @title Update a \code{felm_formula} object
#' @description
#' S3 method for [update()] on objects created with [felm_fml()].  Splits the
#' formula on `|`, updates each segment independently, and reassembles.
#'
#' The `.` placeholder behaves as in [update.formula()]:
#' * `. ~ .` — keep current response and RHS unchanged.
#' * Second `|` segment — replaces (or keeps with `.`) the fixed effects.
#' * Third `|` segment — replaces (or keeps with `.`) the cluster variables.
#'
#' @param object A `felm_formula` object (created by [felm_fml()]).
#' @param formula. Update specification, e.g. `. ~ . | . | ctry1 + ctry2`.
#' @param ... Ignored.
#'
#' @return A new `felm_formula` object.
#' @export
update.felm_formula <- function(object, formula., ...) {
  old <- felm_formula_parts_(object)

  new_chr  <- deparse1(formula.)
  new_segs <- trimws(strsplit(new_chr, "\\|")[[1L]])

  # base (lhs ~ rhs) — safe to delegate to update.formula()
  new_base <- deparse1(
    update.formula(as.formula(old$base), as.formula(new_segs[[1L]]))
  )

  # FE segment
  new_fe <- if (length(new_segs) >= 2L) {
    seg <- new_segs[[2L]]
    if (seg == ".") old$fe else if (nzchar(seg)) seg else NULL
  } else {
    old$fe
  }

  # cluster segment
  new_cluster <- if (length(new_segs) >= 3L) {
    seg <- new_segs[[3L]]
    if (seg == ".") old$cluster else if (nzchar(seg)) seg else NULL
  } else {
    old$cluster
  }

  felm_fml(felm_formula(new_base, fe = new_fe, cluster = new_cluster))
}

#' @title Build a felm / feglm formula from parts
#' @description
#' Constructs a `|`-separated formula accepted by [felm()] and [feglm()] from
#' a base `y ~ x` formula plus optional fixed-effects and cluster components
#' supplied as plain character strings.  This avoids calling [update.formula()],
#' which cannot handle the `|` syntax and raises *"+ not meaningful for
#' factors"* errors.
#'
#' @param base A one-sided or two-sided `formula` object **or** a character
#'   string that can be coerced to one, e.g.
#'   `"ltrade ~ bothin + onein + gsp"`.
#' @param fe Optional character string naming the fixed-effect variable(s),
#'   e.g. `"year"` or `"country + year"`.  Omit or pass `NULL` for models
#'   without fixed effects.
#' @param cluster Optional character string naming the cluster variable(s),
#'   e.g. `"pair"` or `"ctry1 + ctry2"`.  Omit or pass `NULL` for models
#'   without explicit clustering.
#'
#' @return A `formula` of the form `y ~ x`, `y ~ x | fe`, or
#'   `y ~ x | fe | cluster` depending on which parts are supplied.
#'
#' @examples
#' rhs <- "ltrade ~ bothin + onein + lrgdp"
#'
#' felm_formula(rhs)
#' #> ltrade ~ bothin + onein + lrgdp
#'
#' felm_formula(rhs, fe = "year")
#' #> ltrade ~ bothin + onein + lrgdp | year
#'
#' felm_formula(rhs, fe = "year", cluster = "ctry1 + ctry2")
#' #> ltrade ~ bothin + onein + lrgdp | year | ctry1 + ctry2
#'
#' @export
felm_formula <- function(base, fe = NULL, cluster = NULL) {
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

  as.formula(parts)
}
