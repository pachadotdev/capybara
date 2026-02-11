#' srr_stats
#' @srrstats {G1.0} Provides modular helper functions for computations in linear models with fixed effects.
#' @noRd
NULL

#' @title Get score matrix
#' @description Computes the score matrix
#' @param object Result list
#' @noRd
get_score_matrix_felm_ <- function(object) {
  # Update weights and dependent variable
  y <- object[["data"]][[1L]]

  # Center regressor matrix (if required)
  if (object[["control"]][["keep_tx"]]) {
    tx <- object[["tx"]]
  } else {
    # Generate flat FE codes to project out the fixed effects
    k_list <- get_index_list_(names(object[["lvls_k"]]), object[["data"]])

    # Extract regressor matrix
    X <- model.matrix(object[["formula"]], object[["data"]], rhs = 1L)[, -1L, drop = FALSE]
    nms_sp <- attr(X, "dimnames")[[2L]]
    attr(X, "dimnames") <- NULL

    # Center variables
    tx <- center_variables_(
      X, object[["weights"]], k_list[["codes"]],
      object[["control"]][["center_tol"]],
      object[["control"]][["iter_center_max"]]
    )
    colnames(tx) <- nms_sp
  }

  # Return score matrix
  tx * (y * object[["weights"]])
}
