# this is not just about speed/memory, but also about obtaining the same
# slopes as in base R

load_all()
library(dplyr)
library(tidyr)
library(janitor)
library(bench)

rm(list = ls())
gc()

# data ----

ch1_application3 <- tradepolicy::agtpa_applications %>%
  clean_names() %>%
  filter(year %in% seq(1986, 2006, 4)) %>%
  mutate(
    exp_year = paste0(exporter, year),
    imp_year = paste0(importer, year),
    year = paste0("intl_border_", year),
    log_trade = log(trade),
    log_dist = log(dist),
    intl_brdr = ifelse(exporter == importer, pair_id, "inter"),
    intl_brdr_2 = ifelse(exporter == importer, 0, 1),
    pair_id_2 = ifelse(exporter == importer, "0-intra", pair_id)
  ) %>%
  spread(year, intl_brdr_2, fill = 0)

ch1_application3 <- ch1_application3 %>%
  group_by(pair_id) %>%
  mutate(sum_trade = sum(trade)) %>%
  ungroup()

# ppml ----

form <- trade ~ 0 + log_dist + cntg + lang + clny +
  rta + exp_year + imp_year

form2 <- trade ~ log_dist + cntg + lang + clny +
  rta | exp_year + imp_year

d <- filter(ch1_application3, importer != exporter)

bench_ppml <- mark(
  fepoisson(form2, data = d)$coefficients["rta"]
)

formula = form2
data = d
weights = NULL
beta.start = NULL
eta.start = NULL
control = NULL
family <- poisson()

check_formula_(formula)
check_data_(data)
check_family_(family)
control <- check_control_(control)
formula <- update_formula_(formula)
lhs <- NA # just to avoid global variable warning
nobs.na <- NA
nobs.full <- NA
model_frame_(data, formula, weights)
check_response_(data, lhs, family)
k.vars <- attr(terms(formula, rhs = 2L), "term.labels")
k <- length(k.vars)
tmp.var <- temp_var_(data)
data <- drop_by_link_type_(data, lhs, family, tmp.var, k.vars, control)
data <- transform_fe_(data, formula, k.vars)
nt <- nrow(data)
nobs <- nobs_(nobs.full, nobs.na, nt)
nms.sp <- NA
p <- NA
model_response_(data, formula)

p

qr_(X, FALSE)$rank
out <- qr(X)
dim(out$qr)

bench_ppml$median
bench_ppml$mem_alloc

# rm(d)

# trade diversion ----

form <- trade ~ 0 + log_dist + cntg + lang + clny +
  rta + exp_year + imp_year + intl_brdr

form2 <- trade ~ log_dist + cntg + lang + clny +
  rta | exp_year + imp_year + intl_brdr

d <- ch1_application3

bench_trade_diversion <- mark(
  round(fepoisson(form2, data = d)$coefficients["rta"], 3)
)

bench_trade_diversion$median
bench_trade_diversion$mem_alloc

# rm(d)

# endogeneity ----

# form <- trade ~ 0 + rta + exp_year + imp_year + pair_id_2
# form2 <- trade ~ rta | exp_year + imp_year + pair_id_2

# d <- filter(ch1_application3, sum_trade > 0)

# bench_endogeneity <- mark(
#   round(fepoisson(form2, data = d)$coefficients["rta"], 3),
#   iterations = 1000L
# )

bench_endogeneity

# rm(d)

# reverse causality ----

# form <- trade ~ 0 + rta + rta_lead4 + exp_year + imp_year + pair_id_2
# form2 <- trade ~ rta + rta_lead4 | exp_year + imp_year + pair_id_2

# d <- filter(ch1_application3, sum_trade > 0)

# bench_reverse_causality <- mark(
#   round(fepoisson(form2, data = d)$coefficients["rta"], 3)
# )

# bench_reverse_causality

# rm(d)

# non-linear/phasing effects ----

# form <- trade ~ 0 + rta + rta_lag4 + rta_lag8 + rta_lag12 +
#   exp_year + imp_year + pair_id_2

# form2 <- trade ~ rta + rta_lag4 + rta_lag8 + rta_lag12 |
#   exp_year + imp_year + pair_id_2

# d <- filter(ch1_application3, sum_trade > 0)

# bench_phasing <- mark(
#   round(fepoisson(form2, data = d)$coefficients["rta"], 3)
# )

# bench_phasing

# rm(d)

# globalization ----

form <- trade ~ 0 + rta + rta_lag4 + rta_lag8 + rta_lag12 +
  intl_border_1986 + intl_border_1990 + intl_border_1994 +
  intl_border_1998 + intl_border_2002 +
  exp_year + imp_year + pair_id_2

form2 <- trade ~ rta + rta_lag4 + rta_lag8 + rta_lag12 +
  intl_border_1986 + intl_border_1990 + intl_border_1994 +
  intl_border_1998 + intl_border_2002 |
  exp_year + imp_year + pair_id_2

d <- filter(ch1_application3, sum_trade > 0)

bench_globalization <- mark(
  round(fepoisson(form2, data = d)$coefficients["rta"], 3)
)

bench_globalization

rm(d, form, ch1_application3)

rm(list = ls())
gc()
