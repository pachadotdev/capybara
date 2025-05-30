# 1: packages ----

library(bench)
library(dplyr)
library(tidyr)
library(janitor)
library(capybara)

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

### 3.2.6: globalization ----

message("=======")
message("GLOBALIZATION")
message("=======")

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
  round(alpaca::feglm(form2, data = d, family = poisson())$coefficients["rta"], 2),
  round(capybara::fepoisson(form2, data = d)$coefficients["rta"], 2),
  round(fixest::fepois(form2, data = d)$coefficients["rta"], 2)
  # round(stats::glm(form, data = d, family = quasipoisson())$coefficients["rta"], 2)
)

saveRDS(bench_globalization, "dev/bench_globalization.rds")

bench_globalization %>%
  mutate(pkg = c("alpaca", "capybara", "fixest")) %>%
  select(pkg, median, mem_alloc) %>%
  as_tibble()
