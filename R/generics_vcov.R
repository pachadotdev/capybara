#' @title Covariance matrix for APEs
#' @description Covariance matrix for the estimator of the
#'  average partial effects from objects returned by \code{\link{apes}}.
#' @param object an object of class \code{"apes"}.
#' @param ... additional arguments.
#' @return A named matrix of covariance estimates.
#' @seealso \code{\link{apes}}
#' @export
#' @noRd
vcov.apes <- function(object, ...) {
  object[["vcov"]]
}

#' @title Covariance matrix for GLMs
#' @description Covariance matrix for the estimator of the structural parameters
#'  from objects returned by \code{\link{feglm}}. The covariance is computed
#' from the hessian, the scores, or a combination of both after convergence.
#' @param object an object of class \code{"feglm"}.
#' @param type the type of covariance estimate required. \code{"hessian"} refers
#'  to the inverse of the negative expected hessian after convergence and is the
#'  default option. \code{"outer.product"} is the outer-product-of-the-gradient
#'  estimator. \code{"sandwich"} is the sandwich estimator (sometimes also
#'  referred as robust estimator), and \code{"clustered"} computes a clustered
#'  covariance matrix given some cluster variables.
#' @param ... additional arguments.
#' @return A named matrix of covariance estimates.
#' @references Cameron, C., J. Gelbach, and D. Miller (2011). "Robust Inference
#'  With Multiway Clustering". Journal of Business & Economic Statistics 29(2).
#' @seealso \code{\link{feglm}}
#' @examples 
#' mod <- fepoisson(
#'  trade ~ log_dist + lang + cntg + clny | exp_year + imp_year | pair,
#'  trade_panel
#' )
#' 
#' round(vcov(mod, type = "clustered"), 5)
#' 
#' @return A named matrix of covariance estimates.
#' 
#' @export
vcov.feglm <- function(
    object,
    type = c("hessian", "outer.product", "sandwich", "clustered"),
    ...) {
  # Check validity of input argument 'type'
  type <- match.arg(type)

  # Extract cluster from formula
  # it is totally fine not to have a cluster variable
  suppressWarnings({
    cl.vars <- attr(terms(object[["formula"]], rhs = 3L), "term.labels")
  })
  k <- length(cl.vars)
  if (isTRUE(k >= 1L) && type != "clustered") {
    type <- "clustered"
    # add overwrite warning msg
    message(
      paste(
        "There are clustering variables in the model formula.",
        "The 'type' argument will be overwritten to 'cluster'."
      )
    )
  }

  # Compute requested type of covariance matrix
  H <- object[["hessian"]]
  p <- ncol(H)
  if (type == "hessian") {
    # If the hessian is invertible, compute its inverse
    V <- try(solve(H), silent = TRUE)
    if (inherits(V, "try-error")) {
      V <- matrix(Inf, p, p)
    }
  } else {
    G <- get_score_matrix_(object)
    if (type == "outer.product") {
      # Check if the OP is invertible and compute its inverse
      V <- try(solve(G), silent = TRUE)
      if (inherits(V, "try-error")) {
        V <- matrix(Inf, p, p)
      }
    } else {
      # Check if the hessian is invertible and compute its inverse
      V <- try(solve(H), silent = TRUE)
      if (inherits(V, "try-error")) {
        V <- matrix(Inf, p, p)
      } else {
        # Compute inner part of the sandwich formula
        if (type == "sandwich") {
          B <- crossprod(G)
        } else {
          if (isFALSE(k >= 1L)) {
            stop(
              paste(
                "No cluster variable was found.",
                "Please specify a cluster variable",
                "in the model formula."
              ),
              call. = FALSE
            )
          }

          D <- try(object[["data"]][, get("cl.vars"), with = FALSE], silent = TRUE)

          if (inherits(D, "try-error")) {
            stop(
              paste(
                "At least one cluster variable was not found.",
                "Ensure to pass vhttps://www.instagram.com/p/C7fss5CCzNL/ariables that are not part of the model",
                "itself, but are required to compute clustered standard errors",
                "to 'feglm'. This can be done via 'formula'. See documentation",
                "for details."
              ),
              call. = FALSE
            )
          }

          # Ensure cluster variables are factors
          D <- mutate(D, across(all_of(cl.vars), check_factor_))

          # Join cluster variables and scores
          sp.vars <- colnames(G)
          G <- cbind(D, G)
          rm(D)

          # Multiway clustering by Cameron, Gelbach, and Miller (2011)
          B <- matrix(0.0, p, p)
          for (i in seq.int(length(cl.vars))) {
            # Compute outer product for all possible combinations
            cl.combn <- combn(cl.vars, i)
            B.r <- matrix(0.0, p, p)
            for (j in seq.int(ncol(cl.combn))) {
              cl <- cl.combn[, j]
              B.r <- B.r + crossprod(
                as.matrix(
                  G %>%
                    group_by(!!sym(cl)) %>%
                    summarise(across(all_of(sp.vars), sum), .groups = "drop") %>%
                    select(-!!sym(cl))
                )
              )
            }

            # Update outer product
            if (i %% 2L) {
              B <- B + B.r
            } else {
              B <- B - B.r
            }
          }
        }

        # Sandwich formula
        V <- V %*% B %*% V
      }
    }
  }

  # Return covariance estimate
  V
}

#' @title Covariance matrix for LMs
#' @description Covariance matrix for the estimator of the structural parameters
#'  from objects returned by \code{\link{felm}}. The covariance is computed
#' from the hessian, the scores, or a combination of both after convergence.
#' @param object an object of class \code{"felm"}.
#' @inherit vcov.feglm
#' @seealso \code{\link{felm}}
#' @export
vcov.felm <- function(
    object,
    type = c("hessian", "outer.product", "sandwich", "clustered"),
    ...) {
  vcov.feglm(object, type)
}
