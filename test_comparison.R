load_all()

library(dplyr)

# Simple test with the benchmark data
ch1_application3 <- tradepolicy::agtpa_applications %>%
  janitor::clean_names() %>%
  dplyr::filter(year %in% seq(1986, 2006, 4)) %>%
  dplyr::mutate(
    exp_year = paste0(exporter, year),
    imp_year = paste0(importer, year),
    year = paste0("intl_border_", year),
    log_trade = log(trade),
    log_dist = log(dist),
    intl_brdr = ifelse(exporter == importer, pair_id, "inter"),
    intl_brdr_2 = ifelse(exporter == importer, 0, 1),
    pair_id_2 = ifelse(exporter == importer, "0-intra", pair_id)
  ) %>%
  tidyr::pivot_wider(
    names_from = year,
    values_from = intl_brdr_2,
    values_fill = 0
  ) %>%
  dplyr::group_by(pair_id) %>%
  dplyr::mutate(sum_trade = sum(trade)) %>%
  dplyr::ungroup()

d <- dplyr::filter(ch1_application3, sum_trade > 0)

form <- trade ~ rta + rta_lag4 + rta_lag8 + rta_lag12 +
  intl_border_1986 + intl_border_1990 + intl_border_1994 +
  intl_border_1998 + intl_border_2002 |
  exp_year + imp_year + pair_id_2

# Compare results
fixest_result <- fixest::fepois(form, data = d)
capybara_result <- capybara::fepoisson(form, data = d)

cat("Fixest rta coefficient:", round(fixest_result$coefficients["rta"], 4), "\n")
cat("Capybara rta coefficient:", round(capybara_result$coefficients["rta"], 4), "\n")
cat("Difference:", round(fixest_result$coefficients["rta"] - capybara_result$coefficients["rta"], 4), "\n")
