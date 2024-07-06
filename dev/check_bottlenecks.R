load_all()
library(dplyr)
library(tidyr)
library(janitor)
library(profvis)

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

# profvis(fepoisson(form2, data = d))

fepoisson(form2, data = d)

load_all()

foo <- function() {
  x <- feglm(
    trade ~ log_dist + lang + cntg + clny | exp_year + imp_year | pair,
    trade_panel,
    family = poisson(link = "log")
  )

  summary(x, "clustered")
}

m <- bench::mark(foo())

m$median
m$mem_alloc