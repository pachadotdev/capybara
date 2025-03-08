load_all()

library(bench)
library(dplyr)
library(tidyr)
library(janitor)

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

form <- trade ~ rta + rta_lead4 | exp_year + imp_year + pair_id_2

d <- filter(ch1_application3, sum_trade > 0)

mark(
  capybara::fepoisson(form, data = d)$coefficients["rta"],
  fixest::fepois(form, data = d)$coefficients["rta"],
  iterations = 10L
)

Rprof("capybara_profile.out")
mod <- capybara::fepoisson(form, data = d)
Rprof(NULL)

profvis::profvis({
  mod <- capybara::fepoisson(form, data = d)
})
