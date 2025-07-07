#' Fast ML-based Poisson regression with fixed effects
#'
#' @description Optimized Poisson GLM using maximum likelihood approach
#' @param formula an object of class \code{"formula"}
#' @param data an object of class \code{"data.frame"}
#' @param weights an optional vector of weights
#' @param control a list of control parameters
#' @param ... additional arguments
#' @return A named list of class \code{"feglm_ml"}
#' @export
fepoisson_ml <- function(formula = NULL, data = NULL, weights = NULL, 
                         control = NULL, ...) {
  feglm_ml(formula = formula, data = data, family = poisson(),
           weights = weights, control = control, ...)
}

#' Fast ML-based Negative Binomial regression with fixed effects
#'
#' @description Optimized Negative Binomial GLM using maximum likelihood approach
#' @param formula an object of class \code{"formula"}
#' @param data an object of class \code{"data.frame"}
#' @param weights an optional vector of weights
#' @param theta dispersion parameter
#' @param control a list of control parameters
#' @param ... additional arguments
#' @return A named list of class \code{"feglm_ml"}
#' @export
fenegbin_ml <- function(formula = NULL, data = NULL, weights = NULL,
                        theta = 1.0, control = NULL, ...) {
  family <- MASS::negative.binomial(theta = theta)
  feglm_ml(formula = formula, data = data, family = family,
           weights = weights, control = control, ...)
}

#' Fast ML-based Logistic regression with fixed effects
#'
#' @description Optimized Logistic GLM using maximum likelihood approach
#' @param formula an object of class \code{"formula"}
#' @param data an object of class \code{"data.frame"}
#' @param weights an optional vector of weights
#' @param control a list of control parameters
#' @param ... additional arguments
#' @return A named list of class \code{"feglm_ml"}
#' @export
felogit_ml <- function(formula = NULL, data = NULL, weights = NULL, 
                       control = NULL, ...) {
  feglm_ml(formula = formula, data = data, family = binomial(),
           weights = weights, control = control, ...)
}

#' Performance comparison between original and ML implementations
#'
#' @description Compare speed and memory usage between capybara and capybara_ml
#' @param formula an object of class \code{"formula"}
#' @param data an object of class \code{"data.frame"}
#' @param family GLM family to test
#' @param weights optional weights
#' @param n_runs number of benchmark runs
#' @return A list with timing and memory comparisons
#' @export
benchmark_ml <- function(formula, data, family = poisson(), weights = NULL, n_runs = 5, iterations = NULL) {
  
  # Handle backwards compatibility with iterations parameter
  if (!is.null(iterations)) {
    n_runs <- iterations
  }
  
  cat("Benchmarking capybara vs capybara_ml\n")
  cat("=====================================\n")
  
  # Benchmark original implementation
  cat("Running original implementation...\n")
  time_orig <- system.time({
    for (i in 1:n_runs) {
      fit_orig <- feglm(formula = formula, data = data, family = family, weights = weights)
    }
  })
  
  # Benchmark ML implementation
  cat("Running ML implementation...\n")
  time_ml <- system.time({
    for (i in 1:n_runs) {
      fit_ml <- feglm_ml(formula = formula, data = data, family = family, weights = weights)
    }
  })
  
  # Calculate speedup
  speedup <- time_orig[["elapsed"]] / time_ml[["elapsed"]]
  
  cat("\nResults:\n")
  cat("--------\n")
  cat(sprintf("Original time: %.3f seconds\n", time_orig[["elapsed"]]))
  cat(sprintf("ML time:       %.3f seconds\n", time_ml[["elapsed"]]))
  cat(sprintf("Speedup:       %.2fx\n", speedup))
  
  # Memory comparison (if available)
  mem_orig <- object.size(fit_orig)
  mem_ml <- object.size(fit_ml)
  mem_ratio <- as.numeric(mem_ml) / as.numeric(mem_orig)
  
  cat(sprintf("Original memory: %s\n", format(mem_orig, units = "auto")))
  cat(sprintf("ML memory:       %s\n", format(mem_ml, units = "auto")))
  cat(sprintf("Memory ratio:    %.2f\n", mem_ratio))
  
  # Coefficient comparison
  coef_diff <- max(abs(coef(fit_orig) - coef(fit_ml)), na.rm = TRUE)
  cat(sprintf("Max coef diff:   %.2e\n", coef_diff))
  
  invisible(list(
    time_orig = time_orig,
    time_ml = time_ml,
    speedup = speedup,
    mem_orig = mem_orig,
    mem_ml = mem_ml,
    mem_ratio = mem_ratio,
    coef_diff = coef_diff,
    fit_orig = fit_orig,
    fit_ml = fit_ml
  ))
}
