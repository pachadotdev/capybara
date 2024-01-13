#' @title
#' Compute covariance matrix after estimating \code{apes}
#' @description
#' \code{\link{vcov.apes}} estimates the covariance matrix for the estimator of the
#' average partial effects from objects returned by \code{\link{apes}}.
#' @param
#' object an object of class \code{"apes"}.
#' @param
#' ... other arguments.
#' @return
#' The function \code{\link{vcov.apes}} returns a named matrix of covariance estimates.
#' @seealso
#' \code{\link{apes}}
#' @export
vcov.apes <- function(object, ...) {
  object[["vcov"]]
}


#' @title
#' Compute covariance matrix after fitting \code{feglm}
#' @description
#' \code{\link{vcov.feglm}} estimates the covariance matrix for the estimator of the
#' structural parameters from objects returned by \code{\link{feglm}}. The covariance is computed
#' from the Hessian, the scores, or a combination of both after convergence.
#' @param
#' object an object of class \code{"feglm"}.
#' @param
#' type the type of covariance estimate required. \code{"hessian"} refers to the inverse
#' of the negative expected Hessian after convergence and is the default option.
#' \code{"outer.product"} is the outer-product-of-the-gradient estimator,
#' \code{"sandwich"} is the sandwich estimator (sometimes also referred as robust estimator),
#' and \code{"clustered"} computes a clustered covariance matrix given some cluster variables.
#' @param
#' cluster a symbolic description indicating the clustering of observations.
#' @param
#' cluster.vars deprecated; use \code{cluster} instead.
#' @param
#' ... other arguments.
#' @details
#' Multi-way clustering is done using the algorithm of Cameron, Gelbach, and Miller (2011). An
#' example is provided in the vignette "Replicating an Empirical Example of International Trade".
#' @return
#' The function \code{\link{vcov.feglm}} returns a named matrix of covariance estimates.
#' @references
#' Cameron, C., J. Gelbach, and D. Miller (2011). "Robust Inference With Multiway Clustering".
#' Journal of Business & Economic Statistics 29(2).
#' @seealso
#' \code{\link{feglm}}
#' @export
vcov.feglm <- function(
    object,
    type = c("hessian", "outer.product", "sandwich", "clustered"),
    cluster = NULL,
    cluster.vars = NULL,
    ...) {
  # Check validity of input argument 'type'
  type <- match.arg(type)

  # 'cluster.vars' is deprecated
  if (!is.null(cluster.vars)) {
    warning("'cluster.vars' is deprecated; please use 'cluster' instead.", call. = FALSE)
    if (!is.character(cluster.vars)) {
      stop("'cluster.vars' has to be a character.", call. = FALSE)
    }
    cluster <- as.formula(paste0("~", paste0(cluster.vars, collapse = "+")))
  }

  # Compute requested type of covariance matrix
  H <- object[["Hessian"]]
  p <- ncol(H)
  if (type == "hessian") {
    # Check if the Hessian is invertible and compute its inverse
    R <- try(chol(H), silent = TRUE)
    if (inherits(R, "try-error")) {
      V <- matrix(Inf, p, p)
    } else {
      V <- chol2inv(R)
    }
  } else {
    G <- getScoreMatrix(object)
    if (type == "outer.product") {
      # Check if the OPG is invertible and compute its inverse
      R <- try(chol(crossprod(G)), silent = TRUE)
      if (inherits(R, "try-error")) {
        V <- matrix(Inf, p, p)
      } else {
        V <- chol2inv(R)
      }
    } else {
      # Check if the Hessian is invertible and compute its inverse
      R <- try(chol(H), silent = TRUE)
      if (inherits(R, "try-error")) {
        V <- matrix(Inf, p, p)
      } else {
        # Extract data
        data <- object[["data"]]

        # Compute the inverse of the empirical Hessian
        A <- chol2inv(R)

        # Compute inner part of the sandwich formula
        if (type == "sandwich") {
          B <- crossprod(G)
        } else {
          # Check validity of input argument 'cluster'
          if (is.null(cluster)) {
            stop("'cluster' has to be specified.", call. = FALSE)
          } else if (!inherits(cluster, "formula")) {
            stop("'cluster' has to be of class formula.", call. = FALSE)
          }

          # Extract cluster variables
          cluster <- Formula(cluster)
          D <- try(data[, all.vars(cluster), with = FALSE], silent = TRUE)
          if (inherits(D, "try-error")) {
            stop(
              paste(
                "At least one cluster variable was not found.",
                "Ensure to pass variables that are not part of the model itself, but are",
                "required to compute clustered standard errors, to 'feglm'.",
                "This can be done via 'formula'. See documentation for details."
              ),
              call. = FALSE
            )
          }

          # Ensure cluster variables are factors
          cl.vars <- names(D)
          D[cl.vars] <- as.data.frame(lapply(D[cl.vars], check_factor_))

          # Join cluster variables and scores
          sp.vars <- colnames(G)
          G <- cbind(D, G)
          rm(D)

          # Multiway clustering by Cameron, Gelbach, and Miller (2011)
          setkeyv(G, cl.vars)
          B <- matrix(0.0, p, p)
          for (i in seq.int(length(cl.vars))) {
            # Compute outer product for all possible combinations
            cl.combn <- combn(cl.vars, i)
            B.r <- matrix(0.0, p, p)
            for (j in seq.int(ncol(cl.combn))) {
              cl <- cl.combn[, j]
              B.r <- B.r + crossprod(
                as.matrix(
                  do.call(
                    rbind,
                    lapply(split(G, G$cl), function(df) colSums(df[sp.vars]))
                  )
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
        V <- A %*% B %*% A
      }
    }
  }

  # Return covariance estimate
  V
}

#' @title
#' Compute covariance matrix after fitting \code{felm}
#' @description
#' \code{\link{vcov.feglm}} estimates the covariance matrix for the estimator of the
#' structural parameters from objects returned by \code{\link{felm}}. The covariance is computed
#' from the Hessian, the scores, or a combination of both after convergence.
#' @param
#' object an object of class \code{"feglm"}.
#' @param
#' type the type of covariance estimate required. \code{"hessian"} refers to the inverse
#' of the negative expected Hessian after convergence and is the default option.
#' \code{"outer.product"} is the outer-product-of-the-gradient estimator,
#' \code{"sandwich"} is the sandwich estimator (sometimes also referred as robust estimator),
#' and \code{"clustered"} computes a clustered covariance matrix given some cluster variables.
#' @param
#' cluster a symbolic description indicating the clustering of observations.
#' @param
#' cluster.vars deprecated; use \code{cluster} instead.
#' @param
#' ... other arguments.
#' @details
#' Multi-way clustering is done using the algorithm of Cameron, Gelbach, and Miller (2011). An
#' example is provided in the vignette "Replicating an Empirical Example of International Trade".
#' @return
#' The function \code{\link{vcov.feglm}} returns a named matrix of covariance estimates.
#' @references
#' Cameron, C., J. Gelbach, and D. Miller (2011). "Robust Inference With Multiway Clustering".
#' Journal of Business & Economic Statistics 29(2).
#' @seealso
#' \code{\link{felm}}
#' @export
vcov.felm <- function(
    object,
    type = c("hessian", "outer.product", "sandwich", "clustered"),
    cluster = NULL,
    ...) {
  vcov.feglm(object, type, cluster, ...)
}
