#' srr_stats (tests)
#' @srrstats {G1.4a} All internal (non-exported) functions should also be
#'  documented in standard [`roxygen2`](https://roxygen2.r-lib.org/) format,
#'  along with a final `@noRd` tag to suppress automatic generation of `.Rd`
#'  files.
#' @noRd
NULL
 
#' @title Checks if the object is an `feglm` object
#' @description Internal check
#' @param object Object to check
#' @param fun Function name (e.g., "apes")
#' @noRd
apes_bias_check_object_ <- function(object, fun) {
  if (is.null(object)) {
    stop("'object' has to be specified.", call. = FALSE)
  } else if (!inherits(object, "feglm")) {
    stop(
      sprintf(
        "'%s' called on a non-'feglm' object.",
        fun
      ),
      call. = FALSE
    )
  }
}

#' @title Checks if the `feglm` object is a binary choice model
#' @description Internal check
#' @param object Object to check
#' @param fun Function name (e.g., "apes")
#' @srrstats {G1.4a} *All internal (non-exported) functions should also be documented in standard [`roxygen2`](https://roxygen2.r-lib.org/) format, along with a final `@noRd` tag to suppress automatic generation of `.Rd` files.*
#' @noRd
apes_bias_check_binary_model_ <- function(family, fun) {
  if (family[["family"]] != "binomial") {
    stop(
      sprintf("'%s' currently only supports binary choice models.", fun),
      call. = FALSE
    )
  }
}

#' @title Checks if the panel structure string is valid
#' @description Internal check
#' @param panel_structure Object to check
#' @param k Number of fixed effects
#' @srrstats {G1.4a} *All internal (non-exported) functions should also be documented in standard [`roxygen2`](https://roxygen2.r-lib.org/) format, along with a final `@noRd` tag to suppress automatic generation of `.Rd` files.*
#' @noRd
apes_bias_check_panel_ <- function(panel_structure, k) {
  if (panel_structure == "classic") {
    if (!(k %in% c(1L, 2L))) {
      stop(
        paste(
          "panel_structure == 'classic' expects a one- or two-way fixed",
          "effect model."
        ),
        call. = FALSE
      )
    }
  } else {
    if (!(k %in% c(2L, 3L))) {
      stop(
        paste(
          "panel_structure == 'network' expects a two- or three-way fixed",
          "effects model."
        ),
        call. = FALSE
      )
    }
  }
}
