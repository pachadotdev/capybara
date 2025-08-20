library(capybara)
library(fixest)
library(tradepolicy)
library(dplyr)
library(tidyr)
library(janitor)

# Reproduce the exact data processing from benchmark
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

form <- trade ~ rta + rta_lag4 + rta_lag8 + rta_lag12 +
  intl_border_1986 + intl_border_1990 + intl_border_1994 +
  intl_border_1998 + intl_border_2002 |
  exp_year + imp_year + pair_id_2

# Compare results
fixest_result <- fixest::fepois(form, data = d)
capybara_result <- capybara::fepoisson(form, data = d)

# Print detailed comparison
cat("=== DETAILED COMPARISON ===\n")
cat("Fixest rta coefficient:", sprintf("%.6f", fixest_result$coefficients["rta"]), "\n")
cat("Capybara rta coefficient:", sprintf("%.6f", capybara_result$coefficients["rta"]), "\n")
cat("Difference:", sprintf("%.6f", fixest_result$coefficients["rta"] - capybara_result$coefficients["rta"]), "\n")
cat("Relative difference:", sprintf("%.2f%%", 100 * (capybara_result$coefficients["rta"] - fixest_result$coefficients["rta"]) / fixest_result$coefficients["rta"]), "\n")

# Check convergence
cat("\n=== CONVERGENCE INFO ===\n")
cat("Fixest iterations:", fixest_result$niter_fit, "\n")
cat("Capybara converged:", capybara_result$conv, "\n")
cat("Capybara iterations:", capybara_result$iter, "\n")
