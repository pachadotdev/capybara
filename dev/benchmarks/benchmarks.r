# 1: packages ----

# pak::local_install("~/scratch/capybara/capybara", upgrade = FALSE)

library(bench)
library(dplyr)
library(tidyr)
library(janitor)

# 3: benchmarks ----

## 3.1: data ----

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

## 3.2: benchmarks ----

message("=======")
message("TRADE DIVERSION")
message("=======")

form <- trade ~ 0 + log_dist + cntg + lang + clny +
  rta + exp_year + imp_year + intl_brdr

form2 <- trade ~ log_dist + cntg + lang + clny +
  rta | exp_year + imp_year + intl_brdr

d <- ch1_application3

bench_trade_diversion <- mark(
  round(alpaca::feglm(form2, data = d, family = poisson())$coefficients["rta"], 2),
  round(capybara::fepoisson(form2, data = d)$coefficients["rta"], 2),
  round(fixest::fepois(form2, data = d)$coefficients["rta"], 2)
)
