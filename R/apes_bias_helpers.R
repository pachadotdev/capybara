#' srr_stats
#' @srrstats {G2.1a} Validates that the input object is of class `feglm`.
#' @srrstats {G5.2a} Provides unique and informative error messages for invalid object types or missing input.
#' @noRd
NULL

#' @title Checks if the object is an `feglm` object
#' @description Internal check
#' @param object Object to check
#' @param fun Function name (e.g., "apes")
#' @noRd
apes_bias_check_object_ <- function(object, fun) {
  if (is.null(object)) {
    stop("'object' has to be specified.", call. = FALSE) # @srrstats {G5.2a}
  } else if (!inherits(object, "feglm")) {
    stop(
      sprintf(
        "'%s' called on a non-'feglm' object.",
        fun
      ),
      call. = FALSE
    ) # @srrstats {G2.1a}, {G5.2a}
  }
}

#' srr_stats
#' @srrstats {G2.1a} Validates that the input `feglm` object represents a binary choice model.
#' @srrstats {G5.2a} Issues a unique and meaningful error if a non-binary model is detected.
#' @noRd
NULL

#' @title Checks if the `feglm` object is a binary choice model
#' @description Internal check
#' @param family Family object to check
#' @param fun Function name (e.g., "apes")
#' @noRd
apes_bias_check_binary_model_ <- function(family, fun) {
  if (family[["family"]] != "binomial") {
    stop(
      sprintf("'%s' currently only supports binary choice models.", fun),
      call. = FALSE
    ) # @srrstats {G2.1a}, {G5.2a}
  }
}

#' srr_stats
#' @srrstats {G2.1a} Validates that the panel structure string matches expected values.
#' @srrstats {G5.2a} Issues specific error messages if the panel structure and number of fixed effects are inconsistent.
#' @srrstats {G2.3a} Validates input arguments using strict conditions for panel structures.
#' @noRd
NULL

#' @title Checks if the panel structure string is valid
#' @description Internal check
#' @param panel_structure String representing the panel structure.
#' @param k Number of fixed effects.
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
      ) # @srrstats {G5.2a}
    }
  } else {
    if (!(k %in% c(2L, 3L))) {
      stop(
        paste(
          "panel_structure == 'network' expects a two- or three-way fixed",
          "effects model."
        ),
        call. = FALSE
      ) # @srrstats {G5.2a}
    }
  }
}
