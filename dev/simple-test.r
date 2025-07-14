load_all()

trade_short <- trade_panel[trade_panel$exp_year == "CAN1994", ]
trade_short <- trade_short[trade_short$trade > 100, ]
trade_short$trade_200 <- ifelse(trade_short$trade >= 200, 1, 0)
trade_short$trade_200_100 <- as.factor(ifelse(trade_short$trade >=
  200, 1, ifelse(trade_short$trade >= 200, 0.5, 0)))
trade_short$trade_1_minus1 <- ifelse(trade_short$trade >= 200, 1,
  -1
)

feglm(
  trade_200 ~ log_dist | rta + cntg + clny + lang,
  data = trade_short,
  family = binomial()
)

fixest::feglm(
  trade_200 ~ log_dist | rta + cntg + clny + lang,
  data = trade_short,
  family = binomial()
)

# debug

formula = trade_200 ~ log_dist | rta + cntg + clny + lang
data <- trade_short
family <- binomial()
control <- NULL
weights <- NULL
beta_start <- NULL
eta_start <- NULL

check_formula_(formula)
check_data_(data)
check_family_(family)
check_control_(control)

lhs <- NA # just to avoid global variable warning
nobs_na <- NA
nobs_full <- NA
weights_vec <- NA
weights_col <- NA

model_frame_(data, formula, weights)

check_response_(data, lhs, family)

k_vars <- suppressWarnings(attr(terms(formula, rhs = 2L), "term.labels"))
if (length(k_vars) < 1L) {
  k_vars <- "missing_fe"
  data[, `:=`("missing_fe", 1L)]
}

tmp_var <- temp_var_(data)

data <- drop_by_link_type_(data, lhs, family, tmp_var, k_vars, control)

data <- transform_fe_(data, formula, k_vars)

nt <- nrow(data)

nms_sp <- NA
p <- NA

model_response_(data, formula)

if (is.null(weights)) {
  wt <- rep(1.0, nt)
} else if (!all(is.na(weights_vec))) {
  # Weights provided as vector
  wt <- weights_vec
  if (length(wt) != nrow(data)) {
    stop("Length of weights vector must equal number of observations.", call. = FALSE)
  }
} else if (!all(is.na(weights_col))) {
  # Weights provided as formula - use the extracted column name
  wt <- data[[weights_col]]
} else {
  # Weights provided as column name
  wt <- data[[weights]]
}

check_weights_(wt)

start_guesses_(beta_start, eta_start, y, x, beta, nt, wt, p, family)

nms_fe <- lapply(data[, .SD, .SDcols = k_vars], levels)
if (length(nms_fe) > 0L) {
  lvls_k <- vapply(nms_fe, length, integer(1))
} else {
  lvls_k <- c("missing_fe" = 1L)
}

if (!any(lvls_k %in% "missing_fe")) {
  k_list <- get_index_list_(k_vars, data)
} else {
  k_list <- list(list(`1` = seq_len(nt) - 1L))
}

if (is.integer(y)) {
  y <- as.numeric(y)
}

fit <- structure(feglm_fit_(
  beta, eta, y, x, wt, 0.0, family[["family"]], control, k_list
), class = "feglm")