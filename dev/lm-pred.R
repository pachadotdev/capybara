> predict.lm
function (object, newdata, se.fit = FALSE, scale = NULL, df = Inf, 
    interval = c("none", "confidence", "prediction"), level = 0.95, 
    type = c("response", "terms"), terms = NULL, na.action = na.pass, 
    pred.var = res.var/weights, weights = 1, rankdeficient = c("warnif", 
        "simple", "non-estim", "NA", "NAwarn"), tol = 1e-06, 
    verbose = FALSE, ...) 
{
    tt <- terms(object)
    if (!inherits(object, "lm")) 
        warning("calling predict.lm(<fake-lm-object>) ...")
    type <- match.arg(type)
    missRdef <- missing(rankdeficient)
    rankdeficient <- match.arg(rankdeficient)
    noData <- (missing(newdata) || is.null(newdata))
    if (noData) {
        mm <- X <- model.matrix(object)
        mmDone <- TRUE
        offset <- object$offset
    }
    else {
        Terms <- delete.response(tt)
        m <- model.frame(Terms, newdata, na.action = na.action, 
            xlev = object$xlevels)
        if (!is.null(cl <- attr(Terms, "dataClasses"))) 
            .checkMFClasses(cl, m)
        X <- model.matrix(Terms, m, contrasts.arg = object$contrasts)
        if (type != "terms") {
            offset <- model.offset(m)
            if (!is.null(addO <- object$call$offset)) {
                addO <- eval(addO, newdata, environment(tt))
                offset <- if (length(offset)) 
                  offset + addO
                else addO
            }
        }
        mmDone <- FALSE
    }
    n <- length(object$residuals)
    p <- object$rank
    p1 <- seq_len(p)
    piv <- if (p) 
        (qrX <- qr.lm(object))$pivot[p1]
    hasNonest <- (p < ncol(X) && !noData)
    nonest <- integer(0L)
    if (hasNonest) {
        msg1 <- gettext("prediction from rank-deficient fit")
        if (rankdeficient == "simple") {
            warning(gettextf("%s; consider predict(., rankdeficient=\"NA\")", 
                msg1), domain = NA)
        }
        else {
            if (verbose) 
                message("lower-rank qr: determining non-estimable cases")
            stopifnot(is.numeric(tol), tol > 0)
            if (!p) 
                qrX <- qr.lm(object)
            tR <- t(qr.R(qrX))
            pp <- nrow(tR)
            if (verbose) 
                cat(sprintf("  n=%d, p=%d < ncol(X)=%d; ncol(tR)=%d <?< pp=%d (=?= n)\n", 
                  n, p, ncol(X), ncol(tR), pp))
            if (ncol(tR) < pp) {
                tR <- cbind(tR, matrix(0, nrow = pp, ncol = pp - 
                  ncol(tR)))
                if (verbose) 
                  cat(sprintf("    new tR: ncol(tR)=%d =!?= %d = pp = nrow(tR)\n", 
                    ncol(tR), pp))
            }
            d <- c(pp, pp)
            tR[.row(d) > p & .row(d) == .col(d)] <- 1
            nbasis <- qr.Q(qr.default(tR))[, (p + 1L):pp, drop = FALSE]
            Xb <- X[, qrX$pivot] %*% nbasis
            norm2 <- function(x) sqrt(sum(x * x))
            Xb.norm <- apply(Xb, 1L, norm2)
            X.norm <- apply(X, 1L, norm2)
            nonest <- which(tol * X.norm <= Xb.norm)
            if (rankdeficient == "warnif" && length(nonest)) 
                warning(gettextf("%s; attr(*, \"non-estim\") has doubtful cases", 
                  msg1), domain = NA)
        }
    }
    beta <- object$coefficients
    if (type != "terms") {
        predictor <- drop(X[, piv, drop = FALSE] %*% beta[piv])
        if (!is.null(offset)) 
            predictor <- predictor + offset
        if (startsWith(rankdeficient, "NA") && length(nonest)) {
            predictor[nonest] <- NA
            if (rankdeficient == "NAwarn") 
                warning(gettextf("%s: NAs produced for non-estimable cases", 
                  msg1), domain = NA)
        }
        else if (rankdeficient == "non-estim" || (hasNonest && 
            length(nonest))) 
            attr(predictor, "non-estim") <- nonest
    }
    interval <- match.arg(interval)
    if (interval == "prediction") {
        if (missing(newdata)) 
            warning("predictions on current data refer to _future_ responses\n")
        if (missing(newdata) && missing(weights)) {
            w <- weights.default(object)
            if (!is.null(w)) {
                weights <- w
                warning("assuming prediction variance inversely proportional to weights used for fitting\n")
            }
        }
        if (!missing(newdata) && missing(weights) && !is.null(object$weights) && 
            missing(pred.var)) 
            warning("Assuming constant prediction variance even though model fit is weighted\n")
        if (inherits(weights, "formula")) {
            if (length(weights) != 2L) 
                stop("'weights' as formula should be one-sided")
            d <- if (noData) 
                model.frame(object)
            else newdata
            weights <- eval(weights[[2L]], d, environment(weights))
        }
    }
    if (se.fit || interval != "none") {
        w <- object$weights
        res.var <- if (is.null(scale)) {
            r <- object$residuals
            rss <- sum(if (is.null(w)) r^2 else r^2 * w)
            df <- object$df.residual
            rss/df
        }
        else scale^2
        if (type != "terms") {
            if (p > 0) {
                XRinv <- if (missing(newdata) && is.null(w)) 
                  qr.Q(qrX)[, p1, drop = FALSE]
                else X[, piv] %*% qr.solve(qr.R(qrX)[p1, p1])
                ip <- drop(XRinv^2 %*% rep(res.var, p))
            }
            else ip <- rep(0, n)
        }
    }
    if (type == "terms") {
        if (!mmDone) {
            mm <- model.matrix(object)
            mmDone <- TRUE
        }
        aa <- attr(mm, "assign")
        ll <- attr(tt, "term.labels")
        hasintercept <- attr(tt, "intercept") > 0L
        if (hasintercept) 
            ll <- c("(Intercept)", ll)
        aaa <- factor(aa, labels = ll)
        asgn <- split(order(aa), aaa)
        if (hasintercept) {
            asgn$"(Intercept)" <- NULL
            avx <- colMeans(mm)
            termsconst <- sum(avx[piv] * beta[piv])
        }
        nterms <- length(asgn)
        if (nterms > 0) {
            predictor <- matrix(ncol = nterms, nrow = NROW(X))
            dimnames(predictor) <- list(rownames(X), names(asgn))
            if (se.fit || interval != "none") {
                ip <- predictor
                Rinv <- qr.solve(qr.R(qr.lm(object))[p1, p1])
            }
            if (hasintercept) 
                X <- sweep(X, 2L, avx, check.margin = FALSE)
            unpiv <- rep.int(0L, NCOL(X))
            unpiv[piv] <- p1
            for (i in seq.int(1L, nterms, length.out = nterms)) {
                iipiv <- asgn[[i]]
                ii <- unpiv[iipiv]
                iipiv[ii == 0L] <- 0L
                predictor[, i] <- if (any(iipiv > 0L)) 
                  X[, iipiv, drop = FALSE] %*% beta[iipiv]
                else 0
                if (se.fit || interval != "none") 
                  ip[, i] <- if (any(iipiv > 0L)) 
                    as.matrix(X[, iipiv, drop = FALSE] %*% Rinv[ii, 
                      , drop = FALSE])^2 %*% rep.int(res.var, 
                      p)
                  else 0
            }
            if (!is.null(terms)) {
                predictor <- predictor[, terms, drop = FALSE]
                if (se.fit) 
                  ip <- ip[, terms, drop = FALSE]
            }
        }
        else {
            predictor <- ip <- matrix(0, n, 0L)
        }
        attr(predictor, "constant") <- if (hasintercept) 
            termsconst
        else 0
    }
    if (interval != "none") {
        tfrac <- qt((1 - level)/2, df)
        hwid <- tfrac * switch(interval, confidence = sqrt(ip), 
            prediction = sqrt(ip + pred.var))
        if (type != "terms") {
            predictor <- cbind(predictor, predictor + hwid %o% 
                c(1, -1))
            colnames(predictor) <- c("fit", "lwr", "upr")
        }
        else {
            if (!is.null(terms)) 
                hwid <- hwid[, terms, drop = FALSE]
            lwr <- predictor + hwid
            upr <- predictor - hwid
        }
    }
    if (se.fit || interval != "none") {
        se <- sqrt(ip)
        if (type == "terms" && !is.null(terms) && !se.fit) 
            se <- se[, terms, drop = FALSE]
    }
    if (missing(newdata) && !is.null(na.act <- object$na.action)) {
        predictor <- napredict(na.act, predictor)
        if (se.fit) 
            se <- napredict(na.act, se)
    }
    if (type == "terms" && interval != "none") {
        if (missing(newdata) && !is.null(na.act)) {
            lwr <- napredict(na.act, lwr)
            upr <- napredict(na.act, upr)
        }
        list(fit = predictor, se.fit = se, lwr = lwr, upr = upr, 
            df = df, residual.scale = sqrt(res.var))
    }
    else if (se.fit) 
        list(fit = predictor, se.fit = se, df = df, residual.scale = sqrt(res.var))
    else predictor
}
<bytecode: 0x55ad5a6b4398>
<environment: namespace:stats>