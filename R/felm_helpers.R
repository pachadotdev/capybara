#' srr_stats
#' @srrstats {G1.0} Provides modular helper functions for computations in linear models with fixed effects.
#' @noRd
NULL

#' @title Get score matrix
#' @description Computes the score matrix
#' @param object Result list
#' @noRd
get_score_matrix_felm_ <- function(object) {
  # Extract required quantities from result list
  control <- object[["control"]]
  data <- object[["data"]]
  w <- object[["weights"]]

  # Update weights and dependent variable
  y <- data[[1L]]

  # Center regressor matrix (if required)
  if (control[["keep_mx"]]) {
    mx <- object[["mx"]]
  } else {
    # Extract additional required quantities from result list
    formula <- object[["formula"]]
    k_vars <- names(object[["lvls_k"]])

    # Generate auxiliary list of indexes to project out the fixed effects
    k_list <- get_index_list_(k_vars, data)

    # Extract regressor matrix
    x <- model.matrix(formula, data, rhs = 1L)[, -1L, drop = FALSE]
    nms_sp <- attr(x, "dimnames")[[2L]]
    attr(x, "dimnames") <- NULL

    # Center variables
    mx <- center_variables_r_(x, w, k_list, control[["center_tol"]], control[["iter_max"]], control[["iter_interrupt"]], control[["iter_ssr"]])
    colnames(mx) <- nms_sp
  }

  # Return score matrix
  mx * (y * w)
}
