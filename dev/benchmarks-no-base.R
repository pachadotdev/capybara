# 1: packages ----

library(bench)
library(dplyr)
library(tidyr)
library(janitor)
library(capybara)

# 2: examples ----

fit <- fepoisson(
  trade ~ log_dist + cntg + lang + clny + rta | exp_year + imp_year,
  data = trade_panel
)

summary(fit)

fit <- fepoisson(
  trade ~ log_dist + cntg + lang + clny + rta | exp_year + imp_year | pair,
  data = trade_panel
)

summary(fit, type = "clustered")

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

### 3.2.1 ppml ----

message("=======")
message("PPML")
message("=======")

form <- trade ~ 0 + log_dist + cntg + lang + clny +
  rta + exp_year + imp_year

form2 <- trade ~ log_dist + cntg + lang + clny +
  rta | exp_year + imp_year

d <- filter(ch1_application3, importer != exporter)

bench_ppml <- mark(
  round(alpaca::feglm(form2, data = d, family = poisson())$coefficients["rta"], 2),
  round(capybara::fepoisson(form2, data = d)$coefficients["rta"], 2),
  round(fixest::fepois(form2, data = d)$coefficients["rta"], 2)
  # round(stats::glm(form, data = d, family = quasipoisson())$coefficients["rta"], 2)
)

saveRDS(bench_ppml, "dev/bench_ppml.rds")

rm(d)

### 3.2.2: trade diversion ----

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
  # round(stats::glm(form, data = d, family = quasipoisson())$coefficients["rta"], 2)
)

saveRDS(bench_trade_diversion, "dev/bench_trade_diversion.rds")

rm(d)

### 3.2.3: endogeneity ----

message("=======")
message("ENDOGENEITY")
message("=======")

form <- trade ~ 0 + rta + exp_year + imp_year + pair_id_2
form2 <- trade ~ rta | exp_year + imp_year + pair_id_2

d <- filter(ch1_application3, sum_trade > 0)

bench_endogeneity <- mark(
  round(alpaca::feglm(form2, data = d, family = poisson())$coefficients["rta"], 2),
  round(capybara::fepoisson(form2, data = d)$coefficients["rta"], 2),
  round(fixest::fepois(form2, data = d)$coefficients["rta"], 2)
  # round(stats::glm(form, data = d, family = quasipoisson())$coefficients["rta"], 2)
)

saveRDS(bench_endogeneity, "dev/bench_endogeneity.rds")

rm(d)

### 3.2.4: reverse causality ----

message("=======")
message("REVERSE CAUSALITY")
message("=======")

form <- trade ~ 0 + rta + rta_lead4 + exp_year + imp_year + pair_id_2
form2 <- trade ~ rta + rta_lead4 | exp_year + imp_year + pair_id_2

d <- filter(ch1_application3, sum_trade > 0)

bench_reverse_causality <- mark(
  round(alpaca::feglm(form2, data = d, family = poisson())$coefficients["rta"], 2),
  round(capybara::fepoisson(form2, data = d)$coefficients["rta"], 2),
  round(fixest::fepois(form2, data = d)$coefficients["rta"], 2)
  # round(stats::glm(form, data = d, family = quasipoisson())$coefficients["rta"], 2)
)

saveRDS(bench_reverse_causality, "dev/bench_reverse_causality.rds")

rm(d)

### 3.2.5: non-linear/phasing effects ----

message("=======")
message("PHASING")
message("=======")

form <- trade ~ 0 + rta + rta_lag4 + rta_lag8 + rta_lag12 +
  exp_year + imp_year + pair_id_2

form2 <- trade ~ rta + rta_lag4 + rta_lag8 + rta_lag12 |
  exp_year + imp_year + pair_id_2

d <- filter(ch1_application3, sum_trade > 0)

bench_phasing <- mark(
  round(alpaca::feglm(form2, data = d, family = poisson())$coefficients["rta"], 2),
  round(capybara::fepoisson(form2, data = d)$coefficients["rta"], 2),
  round(fixest::fepois(form2, data = d)$coefficients["rta"], 2)
  # round(stats::glm(form, data = d, family = quasipoisson())$coefficients["rta"], 2)
)

saveRDS(bench_phasing, "dev/bench_phasing.rds")

rm(d)

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

### 3.2.7: create tables ----

speed_table <- bench_ppml %>%
  # mutate(package = c("Alpaca", "Capybara", "Fixest", "Base R")) %>%
  mutate(package = c("Alpaca", "Capybara", "Fixest")) %>%
  mutate(model = "PPML") %>%
  select(model, package, median) %>%
  pivot_wider(names_from = model, values_from = median) %>%
  left_join(
    bench_trade_diversion %>%
      # mutate(package = c("Alpaca", "Capybara", "Fixest", "Base R")) %>%
      mutate(package = c("Alpaca", "Capybara", "Fixest")) %>%
      mutate(model = "Trade Diversion") %>%
      select(model, package, median) %>%
      pivot_wider(names_from = model, values_from = median)
  ) %>%
  left_join(
    bench_endogeneity %>%
      # mutate(package = c("Alpaca", "Capybara", "Fixest", "Base R")) %>%
      mutate(package = c("Alpaca", "Capybara", "Fixest")) %>%
      mutate(model = "Endogeneity") %>%
      select(model, package, median) %>%
      pivot_wider(names_from = model, values_from = median)
  ) %>%
  left_join(
    bench_reverse_causality %>%
      # mutate(package = c("Alpaca", "Capybara", "Fixest", "Base R")) %>%
      mutate(package = c("Alpaca", "Capybara", "Fixest")) %>%
      mutate(model = "Reverse Causality") %>%
      select(model, package, median) %>%
      pivot_wider(names_from = model, values_from = median)
  ) %>%
  left_join(
    bench_phasing %>%
      # mutate(package = c("Alpaca", "Capybara", "Fixest", "Base R")) %>%
      mutate(package = c("Alpaca", "Capybara", "Fixest")) %>%
      mutate(model = "Non-linear/Phasing Effects") %>%
      select(model, package, median) %>%
      pivot_wider(names_from = model, values_from = median)
  ) %>%
  left_join(
    bench_globalization %>%
      # mutate(package = c("Alpaca", "Capybara", "Fixest", "Base R")) %>%
      mutate(package = c("Alpaca", "Capybara", "Fixest")) %>%
      mutate(model = "Globalization") %>%
      select(model, package, median) %>%
      pivot_wider(names_from = model, values_from = median)
  ) %>%
  arrange(package)

saveRDS(speed_table, "dev/speed_table.rds")

memory_table <- bench_ppml %>%
  # mutate(package = c("Alpaca", "Capybara", "Fixest", "Base R")) %>%
  mutate(package = c("Alpaca", "Capybara", "Fixest")) %>%
  mutate(model = "PPML") %>%
  select(model, package, mem_alloc) %>%
  pivot_wider(names_from = model, values_from = mem_alloc) %>%
  left_join(
    bench_trade_diversion %>%
      # mutate(package = c("Alpaca", "Capybara", "Fixest", "Base R")) %>%
      mutate(package = c("Alpaca", "Capybara", "Fixest")) %>%
      mutate(model = "Trade Diversion") %>%
      select(model, package, mem_alloc) %>%
      pivot_wider(names_from = model, values_from = mem_alloc)
  ) %>%
  left_join(
    bench_endogeneity %>%
      # mutate(package = c("Alpaca", "Capybara", "Fixest", "Base R")) %>%
      mutate(package = c("Alpaca", "Capybara", "Fixest")) %>%
      mutate(model = "Endogeneity") %>%
      select(model, package, mem_alloc) %>%
      pivot_wider(names_from = model, values_from = mem_alloc)
  ) %>%
  left_join(
    bench_reverse_causality %>%
      # mutate(package = c("Alpaca", "Capybara", "Fixest", "Base R")) %>%
      mutate(package = c("Alpaca", "Capybara", "Fixest")) %>%
      mutate(model = "Reverse Causality") %>%
      select(model, package, mem_alloc) %>%
      pivot_wider(names_from = model, values_from = mem_alloc)
  ) %>%
  left_join(
    bench_phasing %>%
      # mutate(package = c("Alpaca", "Capybara", "Fixest", "Base R")) %>%
      mutate(package = c("Alpaca", "Capybara", "Fixest")) %>%
      mutate(model = "Non-linear/Phasing Effects") %>%
      select(model, package, mem_alloc) %>%
      pivot_wider(names_from = model, values_from = mem_alloc)
  ) %>%
  left_join(
    bench_globalization %>%
      # mutate(package = c("Alpaca", "Capybara", "Fixest", "Base R")) %>%
      mutate(package = c("Alpaca", "Capybara", "Fixest")) %>%
      mutate(model = "Globalization") %>%
      select(model, package, mem_alloc) %>%
      pivot_wider(names_from = model, values_from = mem_alloc)
  ) %>%
  arrange(package)

saveRDS(memory_table, "dev/memory_table.rds")

colnames(speed_table)[1] <- "Package"
speed_table$PPML <- bench::as_bench_time(speed_table$PPML)
speed_table$`Trade Diversion` <- bench::as_bench_time(speed_table$`Trade Diversion`)
speed_table$`Endogeneity` <- bench::as_bench_time(speed_table$`Endogeneity`)
speed_table$`Reverse Causality` <- bench::as_bench_time(speed_table$`Reverse Causality`)
speed_table$`Non-linear/Phasing Effects` <- bench::as_bench_time(speed_table$`Non-linear/Phasing Effects`)

colnames(memory_table)[1] <- "Package"
memory_table$PPML <- bench::as_bench_bytes(memory_table$PPML)
memory_table$`Trade Diversion` <- bench::as_bench_bytes(memory_table$`Trade Diversion`)
memory_table$`Endogeneity` <- bench::as_bench_bytes(memory_table$`Endogeneity`)
memory_table$`Reverse Causality` <- bench::as_bench_bytes(memory_table$`Reverse Causality`)
memory_table$`Non-linear/Phasing Effects` <- bench::as_bench_bytes(memory_table$`Non-linear/Phasing Effects`)

bench_table <- speed_table %>%
  pivot_longer(cols = -Package, names_to = "Model", values_to = "Median Time") %>%
  select(Model, Package, `Median Time`) %>%
  group_by(Model) %>%
  mutate(speedres = dense_rank(`Median Time`)) %>%
  ungroup() %>%
  left_join(
    memory_table %>%
      pivot_longer(cols = -Package, names_to = "Model", values_to = "Memory") %>%
      select(Package, Model, Memory) %>%
      group_by(Model) %>%
      mutate(memres = dense_rank(Memory)) %>%
      ungroup(),
    by = c("Package", "Model")
  ) %>%
  mutate(
    Model = gsub("Non-linear/", "", Model),
    Model = paste(1:6, Model)
  ) %>%
  arrange(Model) %>%
  mutate(
    `Median Time` = as.character(`Median Time`),
    Memory = as.character(Memory)
  ) %>%
  mutate(
    # replace number + s with number + " " + s etc.
    `Median Time` = gsub(" - $", "", paste(gsub("([0-9]+)([a-z])", "\\1 \\2", `Median Time`), speedres, sep = " - ")),
    Memory = gsub(" - $", "", paste(gsub("([0-9]+)([A-Z])", "\\1 \\2", Memory), memres, sep = " - ")),
    Model = gsub("[0-9]+ ", "", Model)
  ) %>%
  select(-speedres, -memres)

saveRDS(bench_table, "dev/bench_table.rds")

knitr::kable(bench_table)
