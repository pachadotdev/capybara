dir.create("benchmarks", recursive = TRUE, showWarnings = FALSE)

sessionInfo()$R.version

# 1: packages ----

library(bench)
library(dplyr)
library(tidyr)
library(janitor)

Sys.setenv(CAPYBARA_ADVANCED_BUILD = "yes")
Sys.setenv(CAPYBARA_NCORES = 4)

devtools::install(upgrade = "never", dependencies = TRUE)

run_base <- F

# ppml, trade diversion, endogeneity, reverse causality, phasing, globalization
run <- c(T, F, F, T, F, T)

# Helper functions for incremental table building ----
init_speed_table <- function() {
  tibble(package = character())
}

init_memory_table <- function() {
  tibble(package = character())
}

update_table <- function(table, bench_data, model_name, metric = "median") {
  packages <- c("Base R", "Alpaca", "Fixest", "Capybara")
  
  new_data <- bench_data %>%
    mutate(package = packages) %>%
    mutate(model = model_name) %>%
    select(model, package, !!sym(metric)) %>%
    pivot_wider(names_from = model, values_from = !!sym(metric))
  
  if (nrow(table) == 0) {
    return(new_data)
  } else {
    return(left_join(table, new_data, by = "package"))
  }
}

# Initialize tables
speed_table <- init_speed_table()
memory_table <- init_memory_table()

# 2: data ----

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

# 3: benchmarks ----

### ppml ----

if (run[1]) {
  message("=======")
  message("PPML")
  message("=======")

  fout <- "benchmarks/bench_ppml.rds"

  if (!file.exists(fout)) {
    d <- filter(ch1_application3, importer != exporter)

    form <- trade ~ 0 + log_dist + cntg + lang + clny + rta + exp_year + imp_year

    if (run_base) {
      bench_ppml <- mark(
        round(stats::glm(form, data = d, family = quasipoisson())$coefficients["rta"], 2),
        iterations = 5L
      )
    }

    form2 <- trade ~ log_dist + cntg + lang + clny + rta | exp_year + imp_year

    bench_ppml_2 <- mark(
      round(alpaca::feglm(form2, data = d, family = poisson())$coefficients["rta"], 2),
      round(fixest::fepois(form2, data = d)$coefficients["rta"], 2),
      round(capybara::fepoisson(form2, data = d)$coefficients["rta"], 2),
      iterations = 25L
    )

    bench_ppml_2 %>%
      select(median, mem_alloc) %>%
      mutate(package = c("alpaca", "fixest", "capybara"))

    bench_ppml <- bench_ppml %>%
      bind_rows(bench_ppml_2)

    rm(d, bench_ppml_2)
  } else {
    bench_ppml <- readRDS(fout)
  }
  
  # Update tables incrementally
  speed_table <- update_table(speed_table, bench_ppml, "PPML", "median")
  memory_table <- update_table(memory_table, bench_ppml, "PPML", "mem_alloc")
  
  # Save intermediate tables
  saveRDS(speed_table, "benchmarks/speed_table_partial.rds")
  saveRDS(memory_table, "benchmarks/memory_table_partial.rds")
}

### trade diversion ----

if (run[2]) {
  message("=======")
  message("TRADE DIVERSION")
  message("=======")

  fout <- "benchmarks/bench_trade_diversion.rds"

  if (!file.exists(fout)) {
    form <- trade ~ 0 + log_dist + cntg + lang + clny + rta + exp_year + imp_year + intl_brdr

    bench_trade_diversion <- mark(
      round(stats::glm(form, data = ch1_application3, family = quasipoisson())$coefficients["rta"], 2),
      iterations = 5L
    )

    form2 <- trade ~ log_dist + cntg + lang + clny + rta | exp_year + imp_year + intl_brdr
    
    # capybara::fepoisson(form2, data = ch1_application3)

    bench_trade_diversion_2 <- mark(
      round(alpaca::feglm(form2, data = ch1_application3, family = poisson())$coefficients["rta"], 2),
      round(fixest::fepois(form2, data = ch1_application3)$coefficients["rta"], 2),
      round(capybara::fepoisson(form2, data = ch1_application3)$coefficients["rta"], 2),
      iterations = 100L
    )

    bench_trade_diversion <- bench_trade_diversion %>%
      bind_rows(bench_trade_diversion_2)

    saveRDS(bench_trade_diversion, fout)

    rm(bench_trade_diversion_2)
  } else {
    bench_trade_diversion <- readRDS(fout)
  }
  
  # Update tables incrementally
  speed_table <- update_table(speed_table, bench_trade_diversion, "Trade Diversion", "median")
  memory_table <- update_table(memory_table, bench_trade_diversion, "Trade Diversion", "mem_alloc")
  
  # Save intermediate tables
  saveRDS(speed_table, "benchmarks/speed_table_partial.rds")
  saveRDS(memory_table, "benchmarks/memory_table_partial.rds")
}

### endogeneity ----

if (run[3]) {
  message("=======")
  message("ENDOGENEITY")
  message("=======")

  fout <- "benchmarks/bench_endogeneity.rds"

  if (!file.exists(fout)) {
    d <- filter(ch1_application3, sum_trade > 0)

    form <- trade ~ 0 + rta + exp_year + imp_year + pair_id_2

    bench_endogeneity <- mark(
      round(stats::glm(form, data = d, family = quasipoisson())$coefficients["rta"], 2),
      iterations = 5L
    )

    form2 <- trade ~ rta | exp_year + imp_year + pair_id_2

    bench_endogeneity_2 <- mark(
      round(alpaca::feglm(form2, data = d, family = poisson())$coefficients["rta"], 2),
      round(fixest::fepois(form2, data = d)$coefficients["rta"], 2),
      round(capybara::fepoisson(form2, data = d)$coefficients["rta"], 2),
      iterations = 100L
    )

    bench_endogeneity <- bench_endogeneity %>%
      bind_rows(bench_endogeneity_2)

    rm(d, bench_endogeneity_2)
  } else {
    bench_endogeneity <- readRDS(fout)
  }
  
  # Update tables incrementally
  speed_table <- update_table(speed_table, bench_endogeneity, "Endogeneity", "median")
  memory_table <- update_table(memory_table, bench_endogeneity, "Endogeneity", "mem_alloc")
  
  # Save intermediate tables
  saveRDS(speed_table, "benchmarks/speed_table_partial.rds")
  saveRDS(memory_table, "benchmarks/memory_table_partial.rds")
}

### reverse causality ----

if (run[4]) {
  message("=======")
  message("REVERSE CAUSALITY")
  message("=======")

  fout <- "benchmarks/bench_reverse_causality.rds"

  if (!file.exists(fout)) {
    d <- filter(ch1_application3, sum_trade > 0)

    form <- trade ~ 0 + rta + rta_lead4 + exp_year + imp_year + pair_id_2

    bench_reverse_causality <- mark(
      round(stats::glm(form, data = d, family = quasipoisson())$coefficients["rta"], 2),
      iterations = 5L
    )

    form2 <- trade ~ rta + rta_lead4 | exp_year + imp_year + pair_id_2

    bench_reverse_causality_2 <- mark(
      round(alpaca::feglm(form2, data = d, family = poisson())$coefficients["rta"], 2),
      round(fixest::fepois(form2, data = d)$coefficients["rta"], 2),
      round(capybara::fepoisson(form2, data = d)$coefficients["rta"], 2),
      iterations = 100L
    )

    bench_reverse_causality <- bench_reverse_causality %>%
      bind_rows(bench_reverse_causality_2)

    rm(d, bench_reverse_causality_2)
  } else {
    bench_reverse_causality <- readRDS(fout)
  }
  
  # Update tables incrementally
  speed_table <- update_table(speed_table, bench_reverse_causality, "Reverse Causality", "median")
  memory_table <- update_table(memory_table, bench_reverse_causality, "Reverse Causality", "mem_alloc")
  
  # Save intermediate tables
  saveRDS(speed_table, "benchmarks/speed_table_partial.rds")
  saveRDS(memory_table, "benchmarks/memory_table_partial.rds")
}

### non-linear/phasing effects ----

if (run[5]) {
  message("=======")
  message("PHASING")
  message("=======")

  fout <- "benchmarks/bench_phasing.rds"

  if (!file.exists(fout)) {
    d <- filter(ch1_application3, sum_trade > 0)

    form <- trade ~ 0 + rta + rta_lag4 + rta_lag8 + rta_lag12 + exp_year + imp_year + pair_id_2

    bench_phasing <- mark(
      round(stats::glm(form, data = d, family = quasipoisson())$coefficients["rta"], 2),
      iterations = 5L
    )

    form2 <- trade ~ rta + rta_lag4 + rta_lag8 + rta_lag12 | exp_year + imp_year + pair_id_2

    bench_phasing_2 <- mark(
      round(alpaca::feglm(form2, data = d, family = poisson())$coefficients["rta"], 2),
      round(fixest::fepois(form2, data = d)$coefficients["rta"], 2),
      round(capybara::fepoisson(form2, data = d, control = capybara::fit_control(method = "aitken"))$coefficients["rta"], 2),
      round(capybara::fepoisson(form2, data = d, control = capybara::fit_control(method = "cg"))$coefficients["rta"], 2),
      round(capybara::fepoisson(form2, data = d, control = capybara::fit_control(method = "hybrid"))$coefficients["rta"], 2),
      round(capybara::fepoisson(form2, data = d, control = capybara::fit_control(method = "kaczmarz"))$coefficients["rta"], 2),
      round(capybara::fepoisson(form2, data = d, control = capybara::fit_control(method = "none"))$coefficients["rta"], 2),
      iterations = 100L
    )

    bench_phasing <- bench_phasing %>%
      bind_rows(bench_phasing_2)

    rm(d, bench_phasing_2)
  } else {
    bench_phasing <- readRDS(fout)
  }
  
  # Update tables incrementally
  speed_table <- update_table(speed_table, bench_phasing, "Non-linear/Phasing Effects", "median")
  memory_table <- update_table(memory_table, bench_phasing, "Non-linear/Phasing Effects", "mem_alloc")
  
  # Save intermediate tables
  saveRDS(speed_table, "benchmarks/speed_table_partial.rds")
  saveRDS(memory_table, "benchmarks/memory_table_partial.rds")
}

### globalization ----

if (run[6]) {
  message("=======")
  message("GLOBALIZATION")
  message("=======")

  fout <- "benchmarks/bench_globalization.rds"

  if (!file.exists(fout)) {
    d <- filter(ch1_application3, sum_trade > 0)

    form <- trade ~ 0 + rta + rta_lag4 + rta_lag8 + rta_lag12 +
      intl_border_1986 + intl_border_1990 + intl_border_1994 +
      intl_border_1998 + intl_border_2002 +
      exp_year + imp_year + pair_id_2

    bench_globalization <- mark(
      round(stats::glm(form, data = d, family = quasipoisson())$coefficients["rta"], 2),
      iterations = 5L
    )

    form2 <- trade ~ rta + rta_lag4 + rta_lag8 + rta_lag12 +
      intl_border_1986 + intl_border_1990 + intl_border_1994 +
      intl_border_1998 + intl_border_2002 |
      exp_year + imp_year + pair_id_2

    bench_globalization_2 <- mark(
      round(alpaca::feglm(form2, data = d, family = poisson())$coefficients["rta"], 2),
      round(fixest::fepois(form2, data = d)$coefficients["rta"], 2),
      round(capybara::fepoisson(form2, data = d)$coefficients["rta"], 2),
      iterations = 100L
    )

    bench_globalization <- bench_globalization %>%
      bind_rows(bench_globalization_2)

    rm(d, bench_globalization_2)
  } else {
    bench_globalization <- readRDS(fout)
  }
  
  # Update tables incrementally
  speed_table <- update_table(speed_table, bench_globalization, "Globalization", "median")
  memory_table <- update_table(memory_table, bench_globalization, "Globalization", "mem_alloc")
  
  # Save intermediate tables
  saveRDS(speed_table, "benchmarks/speed_table_partial.rds")
  saveRDS(memory_table, "benchmarks/memory_table_partial.rds")
}

# Final table saves
saveRDS(speed_table, "benchmarks/speed_table.rds")
saveRDS(memory_table, "benchmarks/memory_table.rds")

# Convert data types for visualization
colnames(speed_table)[1] <- "package"
if ("PPML" %in% colnames(speed_table)) speed_table$PPML <- bench::as_bench_time(speed_table$PPML)
if ("Trade Diversion" %in% colnames(speed_table)) speed_table$`Trade Diversion` <- bench::as_bench_time(speed_table$`Trade Diversion`)
if ("Endogeneity" %in% colnames(speed_table)) speed_table$`Endogeneity` <- bench::as_bench_time(speed_table$`Endogeneity`)
if ("Reverse Causality" %in% colnames(speed_table)) speed_table$`Reverse Causality` <- bench::as_bench_time(speed_table$`Reverse Causality`)
if ("Non-linear/Phasing Effects" %in% colnames(speed_table)) speed_table$`Non-linear/Phasing Effects` <- bench::as_bench_time(speed_table$`Non-linear/Phasing Effects`)
if ("Globalization" %in% colnames(speed_table)) speed_table$`Globalization` <- bench::as_bench_time(speed_table$`Globalization`)

colnames(memory_table)[1] <- "package"
if ("PPML" %in% colnames(memory_table)) memory_table$PPML <- bench::as_bench_bytes(memory_table$PPML)
if ("Trade Diversion" %in% colnames(memory_table)) memory_table$`Trade Diversion` <- bench::as_bench_bytes(memory_table$`Trade Diversion`)
if ("Endogeneity" %in% colnames(memory_table)) memory_table$`Endogeneity` <- bench::as_bench_bytes(memory_table$`Endogeneity`)
if ("Reverse Causality" %in% colnames(memory_table)) memory_table$`Reverse Causality` <- bench::as_bench_bytes(memory_table$`Reverse Causality`)
if ("Non-linear/Phasing Effects" %in% colnames(memory_table)) memory_table$`Non-linear/Phasing Effects` <- bench::as_bench_bytes(memory_table$`Non-linear/Phasing Effects`)
if ("Globalization" %in% colnames(memory_table)) memory_table$`Globalization` <- bench::as_bench_bytes(memory_table$`Globalization`)

# Create final benchmark table
bench_table <- speed_table %>%
  pivot_longer(cols = -package, names_to = "Model", values_to = "Median Time") %>%
  select(Model, package, `Median Time`) %>%
  group_by(Model) %>%
  mutate(speedres = dense_rank(`Median Time`)) %>%
  ungroup() %>%
  left_join(
    memory_table %>%
      pivot_longer(cols = -package, names_to = "Model", values_to = "Memory") %>%
      select(package, Model, Memory) %>%
      group_by(Model) %>%
      mutate(memres = dense_rank(Memory)) %>%
      ungroup(),
    by = c("package", "Model")
  ) %>%
  mutate(
    Model = gsub("Non-linear/", "", Model),
    Model = paste(1:length(unique(Model)), Model)
  ) %>%
  arrange(Model) %>%
  mutate(
    `Median Time` = as.character(`Median Time`),
    Memory = as.character(Memory)
  ) %>%
  mutate(
    `Median Time` = gsub(" - $", "", paste(gsub("([0-9]+)([a-z])", "\\1 \\2", `Median Time`), speedres, sep = " - ")),
    Memory = gsub(" - $", "", paste(gsub("([0-9]+)([A-Z])", "\\1 \\2", Memory), memres, sep = " - ")),
    Model = gsub("[0-9]+ ", "", Model)
  ) %>%
  select(-speedres, -memres)

bench_table

saveRDS(bench_table, "benchmarks/bench_table.rds")
