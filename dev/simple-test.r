devtools::load_all()

# ok, no fixed effects works well
mod1 <- felm(mpg ~ wt, mtcars)
mod1$coefficients
mod1$fixed.effects
lm(mpg ~ wt, mtcars)

# ok, it works ok with one fixed effect
mod2 <- felm(mpg ~ wt | cyl, mtcars)
mod2$coefficients
mod2$fixed.effects
lm(mpg ~ wt + as.factor(cyl), mtcars)

mod2$fixed.effects[[1]][1]
mod2$fixed.effects[[1]][2] - mod2$fixed.effects[[1]][1]
mod2$fixed.effects[[1]][3] - mod2$fixed.effects[[1]][1]

# not working with two fixed effects
mod3 <- felm(mpg ~ wt | cyl + am, mtcars)
mod3$coefficients
mod3$fixed.effects

mod3_fixest <- fixest::feols(mpg ~ wt | cyl + am, mtcars)
mod3_fixest$coefficients
fixest::fixef(mod3_fixest)

lm(mpg ~ wt + as.factor(cyl) + as.factor(am), mtcars)

formula = mpg ~ wt | cyl + am
data <- mtcars
weights <- NULL
control <- NULL

# Check validity of formula ----
check_formula_(formula)

# Check validity of data ----
check_data_(data)

# Check validity of control + Extract control list ----
check_control_(control)

# Generate model.frame
lhs <- NA # just to avoid global variable warning
nobs_na <- NA
nobs_full <- NA
weights_vec <- NA
weights_col <- NA
model_frame_(data, formula, weights)

# Get names of the fixed effects variables and sort ----
# the no FEs warning is printed in the check_formula_ function
k_vars <- suppressWarnings(attr(terms(formula, rhs = 2L), "term.labels"))
if (length(k_vars) < 1L) {
  k_vars <- "missing_fe"
  data[, `:=`("missing_fe", 1L)]
}

# Generate temporary variable ----
tmp_var <- temp_var_(data)

# Transform fixed effects and clusters to factors ----
data <- transform_fe_(data, formula, k_vars)

# Determine the number of dropped observations ----
nt <- nrow(data)

# Extract model response and regressor matrix ----
nms_sp <- NA
p <- NA
model_response_(data, formula)

# Extract weights if required ----
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

# Check validity of weights ----
check_weights_(wt)

# Get names and number of levels in each fixed effects category ----
nms_fe <- lapply(data[, .SD, .SDcols = k_vars], levels)
if (length(nms_fe) > 0L) {
  lvls_k <- vapply(nms_fe, length, integer(1))
} else {
  lvls_k <- c("missing_fe" = 1L)
}

# Generate auxiliary list of indexes for different sub panels ----
if (!any(lvls_k %in% "missing_fe")) {
  k_list <- get_index_list_(k_vars, data)
} else {
  k_list <- list(list(`1` = seq_len(nt) - 1L))
}

# Fit linear model ----
if (is.integer(y)) {
  y <- as.numeric(y)
}
fit <- structure(felm_fit_(y, x, wt, control, k_list), class = "felm")
