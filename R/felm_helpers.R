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
  if (control[["keep_tx"]]) {
    tx <- object[["tx"]]
  } else {
    # Extract additional required quantities from result list
    formula <- object[["formula"]]
    k_vars <- names(object[["lvls_k"]])

    # Generate auxiliary list of indexes to project out the fixed effects
    k_list <- get_index_list_(k_vars, data)

    # Extract regressor matrix
    X <- model.matrix(formula, data, rhs = 1L)[, -1L, drop = FALSE]
    nms_sp <- attr(X, "dimnames")[[2L]]
    attr(X, "dimnames") <- NULL

    # Center variables
    defaults <- fit_control()
    get_param <- function(name) {
      if (is.null(control[[name]])) defaults[[name]] else control[[name]]
    }
    
    tx <- center_variables_(X, w, k_list, 
                           control[["center_tol"]], 
                           control[["iter_max"]], 
                           control[["iter_interrupt"]], 
                           control[["iter_ssr"]], 
                           control[["accel_start"]], 
                           get_param("project_tol_factor"), 
                           get_param("grand_accel_tol"), 
                           get_param("project_group_tol"), 
                           get_param("irons_tuck_tol"), 
                           get_param("grand_accel_interval"), 
                           get_param("irons_tuck_interval"), 
                           get_param("ssr_check_interval"), 
                           get_param("convergence_factor"), 
                           get_param("tol_multiplier"))
    colnames(tx) <- nms_sp
  }

  # Return score matrix
  tx * (y * w)
}
