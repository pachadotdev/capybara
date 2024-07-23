# this is not just about speed/memory, but also about obtaining the same
# slopes as in base R

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
  round(alpaca::feglm(form2, data = d, family = poisson())$coefficients["rta"], 3)
)

saveRDS(bench_ppml, "dev/bench_ppml.rds")

rm(d)

# trade diversion ----

form <- trade ~ 0 + log_dist + cntg + lang + clny +
  rta + exp_year + imp_year + intl_brdr

form2 <- trade ~ log_dist + cntg + lang + clny +
  rta | exp_year + imp_year + intl_brdr

d <- ch1_application3

bench_trade_diversion <- mark(
  round(alpaca::feglm(form2, data = d, family = poisson())$coefficients["rta"], 3)
)

saveRDS(bench_trade_diversion, "dev/bench_trade_diversion.rds")

rm(d)

# endogeneity ----

form <- trade ~ 0 + rta + exp_year + imp_year + pair_id_2
form2 <- trade ~ rta | exp_year + imp_year + pair_id_2

d <- filter(ch1_application3, sum_trade > 0)

bench_endogeneity <- mark(
  round(alpaca::feglm(form2, data = d, family = poisson())$coefficients["rta"], 3)
)

saveRDS(bench_endogeneity, "dev/bench_endogeneity.rds")

rm(d)

# reverse causality ----

form <- trade ~ 0 + rta + rta_lead4 + exp_year + imp_year + pair_id_2
form2 <- trade ~ rta + rta_lead4 | exp_year + imp_year + pair_id_2

d <- filter(ch1_application3, sum_trade > 0)

bench_reverse_causality <- mark(
  round(alpaca::feglm(form2, data = d, family = poisson())$coefficients["rta"], 3)
)

saveRDS(bench_reverse_causality, "dev/bench_reverse_causality.rds")

rm(d)

# non-linear/phasing effects ----

form <- trade ~ 0 + rta + rta_lag4 + rta_lag8 + rta_lag12 +
  exp_year + imp_year + pair_id_2

form2 <- trade ~ rta + rta_lag4 + rta_lag8 + rta_lag12 |
  exp_year + imp_year + pair_id_2

d <- filter(ch1_application3, sum_trade > 0)

bench_phasing <- mark(
  round(alpaca::feglm(form2, data = d, family = poisson())$coefficients["rta"], 3)
)

saveRDS(bench_phasing, "dev/bench_phasing.rds")

rm(d)

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
  round(alpaca::feglm(form2, data = d, family = poisson())$coefficients["rta"], 3)
)

saveRDS(bench_globalization, "dev/bench_globalization.rds")

rm(d, form, ch1_application3)

rm(list = ls())
gc()

# create tables ----

library(knitr)

bench_ppml <- readRDS("dev/bench_ppml.rds")

bench_trade_diversion <- readRDS("dev/bench_trade_diversion.rds")

bench_endogeneity <- readRDS("dev/bench_endogeneity.rds")

bench_reverse_causality <- readRDS("dev/bench_reverse_causality.rds")

bench_phasing <- readRDS("dev/bench_phasing.rds")

bench_globalization <- readRDS("dev/bench_globalization.rds")

bench_ppml %>%
  # mutate(package = c("Base R", "**Capybara**", "Fixest", "Alpaca")) %>%
  mutate(package = c("Alpaca")) %>%
  # mutate(package = c("**Capybara**", "Fixest", "Alpaca")) %>%
  mutate(model = "PPML") %>%
  select(model, package, median) %>%
  pivot_wider(names_from = model, values_from = median) %>%
  left_join(
    bench_trade_diversion %>%
      # mutate(package = c("Base R", "**Capybara**", "Fixest", "Alpaca")) %>%
      mutate(package = c("Alpaca")) %>%
      mutate(model = "Trade Diversion") %>%
      select(model, package, median) %>%
      pivot_wider(names_from = model, values_from = median)
  ) %>%
  left_join(
    bench_endogeneity %>%
      # mutate(package = c("Base R", "**Capybara**", "Fixest", "Alpaca")) %>%
      mutate(package = c("Alpaca")) %>%
      mutate(model = "Endogeneity") %>%
      select(model, package, median) %>%
      pivot_wider(names_from = model, values_from = median)
  ) %>%
  left_join(
    bench_reverse_causality %>%
      # mutate(package = c("Base R", "**Capybara**", "Fixest", "Alpaca")) %>%
      mutate(package = c("Alpaca")) %>%
      mutate(model = "Reverse Causality") %>%
      select(model, package, median) %>%
      pivot_wider(names_from = model, values_from = median)
  ) %>%
  left_join(
    bench_phasing %>%
      # mutate(package = c("Base R", "**Capybara**", "Fixest", "Alpaca")) %>%
      mutate(package = c("Alpaca")) %>%
      mutate(model = "Non-linear/Phasing Effects") %>%
      select(model, package, median) %>%
      pivot_wider(names_from = model, values_from = median)
  ) %>%
  left_join(
    bench_globalization %>%
      # mutate(package = c("Base R", "**Capybara**", "Fixest", "Alpaca")) %>%
      mutate(package = c("Alpaca")) %>%
      mutate(model = "Globalization") %>%
      select(model, package, median) %>%
      pivot_wider(names_from = model, values_from = median)
  ) %>%
  arrange(package) %>%
  kable()

bench_ppml %>%
  # mutate(package = c("Base R", "**Capybara**", "Fixest", "Alpaca")) %>%
  mutate(package = c("Alpaca")) %>%
  mutate(model = "PPML") %>%
  select(model, package, mem_alloc) %>%
  pivot_wider(names_from = model, values_from = mem_alloc) %>%
  left_join(
    bench_trade_diversion %>%
      # mutate(package = c("Base R", "**Capybara**", "Fixest", "Alpaca")) %>%
      mutate(package = c("Alpaca")) %>%
      mutate(model = "Trade Diversion") %>%
      select(model, package, mem_alloc) %>%
      pivot_wider(names_from = model, values_from = mem_alloc)
  ) %>%
  left_join(
    bench_endogeneity %>%
      # mutate(package = c("Base R", "**Capybara**", "Fixest", "Alpaca")) %>%
      mutate(package = c("Alpaca")) %>%
      mutate(model = "Endogeneity") %>%
      select(model, package, mem_alloc) %>%
      pivot_wider(names_from = model, values_from = mem_alloc)
  ) %>%
  left_join(
    bench_reverse_causality %>%
      # mutate(package = c("Base R", "**Capybara**", "Fixest", "Alpaca")) %>%
      mutate(package = c("Alpaca")) %>%
      mutate(model = "Reverse Causality") %>%
      select(model, package, mem_alloc) %>%
      pivot_wider(names_from = model, values_from = mem_alloc)
  ) %>%
  left_join(
    bench_phasing %>%
      # mutate(package = c("Base R", "**Capybara**", "Fixest", "Alpaca")) %>%
      mutate(package = c("Alpaca")) %>%
      mutate(model = "Non-linear/Phasing Effects") %>%
      select(model, package, mem_alloc) %>%
      pivot_wider(names_from = model, values_from = mem_alloc)
  ) %>%
  left_join(
    bench_globalization %>%
      # mutate(package = c("Base R", "**Capybara**", "Fixest", "Alpaca")) %>%
      mutate(package = c("Alpaca")) %>%
      mutate(model = "Globalization") %>%
      select(model, package, mem_alloc) %>%
      pivot_wider(names_from = model, values_from = mem_alloc)
  ) %>%
  arrange(package) %>%
  kable()
