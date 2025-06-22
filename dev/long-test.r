# packages ----

if (!require("lfe")) { install.packages("lfe") }

if (!require("alpaca")) { install.packages("alpaca") }

if (!require("fixest")) { install.packages("fixest") }

library(bench)
library(dplyr)
library(tidyr)
library(janitor)

Sys.setenv(CAPYBARA_ADVANCED_BUILD = "yes")
Sys.setenv(CAPYBARA_NCORES = 4)

devtools::install(upgrade = "never", dependencies = TRUE)

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
  pivot_wider(
    names_from = year,
    values_from = intl_brdr_2,
    values_fill = 0
  ) %>%
  group_by(pair_id) %>%
  mutate(sum_trade = sum(trade)) %>%
  ungroup()

# ols ----

d <- filter(ch1_application3, trade > 0)

form <- log_trade ~ log_dist + cntg + lang + clny + rta | exp_year + imp_year

m1 <- as.numeric(lfe::felm(form, data = d)$coefficients)[5]
m2 <- fixest::feols(form, data = d)$coefficients["rta"]
m3 <- capybara::felm(form, data = d)$coefficients["rta"]

all.equal(m1, unname(m3))
all.equal(m2, m3)

# ppml ----

d <- filter(ch1_application3, importer != exporter)

form <- trade ~ log_dist + cntg + lang + clny + rta | exp_year + imp_year

m1 <- alpaca::feglm(form, data = d, family = poisson())$coefficients["rta"]
m2 <- fixest::fepois(form, data = d)$coefficients["rta"]
m3 <- capybara::fepoisson(form, data = d)$coefficients["rta"]

all.equal(m1, m3)
all.equal(m2, m3)

# trade diversion ----

form <- trade ~ log_dist + cntg + lang + clny + rta | exp_year + imp_year + intl_brdr

m1 <- alpaca::feglm(form, data = ch1_application3, family = poisson())$coefficients["rta"]
m2 <- fixest::fepois(form, data = ch1_application3)$coefficients["rta"]
m3 <- capybara::fepoisson(form, data = ch1_application3)$coefficients["rta"]

all.equal(m1, m3)
all.equal(m2, m3)

# [1] "Mean relative difference: 7.640188e-07"

#  endogeneity ----

d <- filter(ch1_application3, sum_trade > 0)

form <- trade ~ rta | exp_year + imp_year + pair_id_2

m1 <- alpaca::feglm(form, data = d, family = poisson())$coefficients["rta"]
m2 <- fixest::fepois(form, data = d)$coefficients["rta"]
m3 <- capybara::fepoisson(form, data = d)$coefficients["rta"]

all.equal(m1, m3)
all.equal(m2, m3)

# reverse causality ----

d <- filter(ch1_application3, sum_trade > 0)

form <- trade ~ rta + rta_lead4 | exp_year + imp_year + pair_id_2

m1 <- alpaca::feglm(form, data = d, family = poisson())$coefficients["rta"]
m2 <- fixest::fepois(form, data = d)$coefficients["rta"]
m3 <- capybara::fepoisson(form, data = d)$coefficients["rta"]

all.equal(m1, m3)
all.equal(m2, m3)

# phasing ----

d <- filter(ch1_application3, sum_trade > 0)

form <- trade ~ rta + rta_lag4 + rta_lag8 + rta_lag12 | exp_year + imp_year + pair_id_2

m1 <- alpaca::feglm(form, data = d, family = poisson())$coefficients["rta"]
m2 <- fixest::fepois(form, data = d)$coefficients["rta"]
m3 <- capybara::fepoisson(form, data = d)$coefficients["rta"]

all.equal(m1, m3)
all.equal(m2, m3)

# globalization ----

d <- filter(ch1_application3, sum_trade > 0)

form2 <- trade ~ rta + rta_lag4 + rta_lag8 + rta_lag12 +
  intl_border_1986 + intl_border_1990 + intl_border_1994 +
  intl_border_1998 + intl_border_2002 |
  exp_year + imp_year + pair_id_2

m1 <- alpaca::feglm(form2, data = d, family = poisson())$coefficients["rta"]
m2 <- fixest::fepois(form2, data = d)$coefficients["rta"]
m3 <- capybara::fepoisson(form2, data = d)$coefficients["rta"]

all.equal(m1, m3)
all.equal(m2, m3)
