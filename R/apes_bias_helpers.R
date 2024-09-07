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

apes_bias_check_binary_model_ <- function(family, fun) {
  if (family[["family"]] != "binomial") {
    stop(
      sprintf("'%s' currently only supports binary choice models.", fun),
      call. = FALSE
    )
  }
}

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
