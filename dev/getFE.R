
  # Extract required quantities from result list
  beta <- m1_lm[["coefficients"]]
  data <- m1_lm[["data"]]
  eta <- m1_lm[["fitted.values"]]
  formula <- m1_lm[["formula"]]
   
  # Extract regressor matrix
  X <- model.matrix(formula, data, rhs = 1L)[, - 1L, drop = FALSE]
  nms.sp <- attr(X, "dimnames")[[2L]]
  attr(X, "dimnames") <- NULL
  
  # Generate auxiliary list of indexes for different sub panels
  fe_vars <- suppressWarnings(attr(terms(formula, rhs = 2L), "term.labels"))
  FEs <- get_index_list_(fe_vars, data)

  # Recover fixed effects by alternating between the solutions of normal equations
  pi <- eta - as.vector(X %*% beta)
  fe.list <- as.list(getAlpha(pi, k.list, alpha.tol))
  
  # Assign names to the different fixed effects categories
  for (i in seq.int(k)) {
    fe.list[[i]] <- as.vector(fe.list[[i]])
    names(fe.list[[i]]) <- nms.fe[[i]]
  }
  names(fe.list) <- k.vars
  
  # Return list of estimated fixed effects
  fe.list
}