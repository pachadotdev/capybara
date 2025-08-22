library(bench)
library(dplyr)
library(tidyr)
library(janitor)

Sys.setenv(CAPYBARA_OPTIMIZATIONS = "yes")
Sys.setenv(CAPYBARA_NCORES = 4)

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
# d <- filter(ch1_application3, trade > 0)

form <- trade ~ log_dist + cntg + lang + clny + rta | exp_year + imp_year

comparable_fixest <- function() {
  out <- list(fit = fixest::fepois(form, data = d))
  out$fixed_effects <- fixest::fixef(out$fit)
  out$fit$coefficients
  

  round(out$fit$coefficients, 2)
}

bench_ppml <- mark(
  comparable_fixest(),
  round(capybara::fepoisson(form, data = d)$coefficients, 2),
  iterations = 20L,
  min_iterations = 20L,
  max_iterations = 20L,
  filter_gc = FALSE
)

print(
  bench_ppml %>%
    mutate(package = c("Fixest", "Capybara")) %>%
    select(package, median, mem_alloc) %>%
    mutate(relative_time = median / lag(median))
)
