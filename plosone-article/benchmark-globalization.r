library(bench)
library(dplyr)
library(tidyr)
library(janitor)

Sys.setenv(CAPYBARA_OPTIMIZATIONS = "yes")
Sys.setenv(CAPYBARA_NCORES = 8)

devtools::install(upgrade = "never", dependencies = TRUE)

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

d <- filter(ch1_application3, sum_trade > 0)

length(unique(d$exp_year)) +
length(unique(d$imp_year)) +
length(unique(d$pair_id_2))

glimpse(d)

form <- trade ~ rta + rta_lag4 + rta_lag8 + rta_lag12 +
  intl_border_1986 + intl_border_1990 + intl_border_1994 +
  intl_border_1998 + intl_border_2002 |
  exp_year + imp_year + pair_id_2

form2 <- trade ~ rta + rta_lag4 + rta_lag8 + rta_lag12 +
  intl_border_1986 + intl_border_1990 + intl_border_1994 +
  intl_border_1998 + intl_border_2002 +
  as.factor(exp_year) + as.factor(imp_year) + as.factor(pair_id_2)

equivalent_full_fepois <- function() {
  out <- fixest::fepois(form, data = d)
  out$fixed_effects <- fixest::fixef(out)
  out
}

# bench_globalization <- mark(
#     round(alpaca::feglm(form, data = d, family = poisson())$coefficients["rta"], 3),
#     round(equivalent_full_fepois()$coefficients["rta"], 3),
#     round(capybara::fepoisson(form, data = d)$coefficients["rta"], 3),
#     iterations = 20L,
#     min_iterations = 20L,
#     max_iterations = 20L,
#     filter_gc = FALSE
#   )

# bench_globalization %>%
#   mutate(package = c("Alpaca", "Fixest", "Capybara")) %>%
#   select(package, median, mem_alloc) %>%
#   mutate(
#     median = round(as.numeric(median), 3),
#     mem_alloc = round(as.numeric(mem_alloc) / 1e6, 3)
#   )

fout <- "dev/article/benchmark-globalization.csv"

if (!file.exists(fout)) {
  bench_globalization <- mark(
    round(alpaca::feglm(form, data = d, family = poisson())$coefficients["rta"], 3),
    round(equivalent_full_fepois()$coefficients["rta"], 3),
    round(capybara::fepoisson(form, data = d)$coefficients["rta"], 3),
    iterations = 20L,
    min_iterations = 20L,
    max_iterations = 20L,
    filter_gc = FALSE
  )

  bench_globalization %>%
    mutate(package = c("Alpaca", "Fixest", "Capybara")) %>%
    select(package, median, mem_alloc) %>%
    mutate(
      median = round(as.numeric(median), 3),
      mem_alloc = round(as.numeric(mem_alloc) / 1e6, 3)
    ) %>%
    readr::write_csv(fout)
}

fout2 <- "dev/article/benchmark-globalization-base.csv"

if (!file.exists(fout2)) {
  bench_globalization_base <- mark(
    round(glm(form2, data = d, family = quasipoisson())$coefficients["rta"], 3),
    iterations = 5L,
    min_iterations = 5L,
    max_iterations = 5L,
    filter_gc = FALSE
  )

  bench_globalization_base %>%
    mutate(package = c("Base R")) %>%
    select(package, median, mem_alloc) %>%
    mutate(
      median = round(as.numeric(median), 3),
      mem_alloc = round(as.numeric(mem_alloc) / 1e6, 3)
    ) %>%
    readr::write_csv(fout2, append = TRUE)
}


if (!file.exists(fout)) {
  bench_globalization <- readr::read_csv(fout, show_col_types = FALSE) %>%
    bind_rows(
      readr::read_csv(fout2, show_col_types = FALSE)
    ) %>%
    mutate(
      rel_median = round(100 * median / max(median), 3),
      rel_mem_alloc = round(100 * mem_alloc / max(mem_alloc), 3)
    )

  readr::write_csv(bench_globalization, fout)
}

unlink(fout2)
