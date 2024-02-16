library(tradepolicy)
library(dplyr)

tradepolicy::agtpa_applications

trade_panel = trade_panel %>%
  as_tibble() %>%
  inner_join(
    tradepolicy::agtpa_applications %>%
      janitor::clean_names() %>%
      select(exporter, importer, year, rta) %>%
      mutate(
        exp_year = paste0(exporter, year),
        imp_year = paste0(importer, year)
      ), by = c("exp_year", "imp_year")
  ) %>%
  select(exp_year, imp_year, trade, dist, cntg, lang, clny, rta) %>%
  mutate(rta = as.integer(rta))

trade_panel = trade_panel %>%
  mutate(
    exporter = substr(exp_year, 1, 3),
    importer = substr(imp_year, 1, 3)
  ) %>%
    filter(importer != exporter) %>%
    select(-exporter, -importer)

use_data(trade_panel, overwrite = T)

load_all()

trade_panel = trade_panel %>%
    mutate(
       exporter = substr(exp_year, 1, 3),
    importer = substr(imp_year, 1, 3),
      pair = paste(exporter, importer, sep = "-"),
      log_dist = log(dist)
      ) %>%
    select(exp_year, imp_year, pair, trade, log_dist, everything()) %>%
        select(-exporter, -importer, -dist)

trade_panel$year = as.integer(substr(trade_panel$exp_year, 4, 7))

trade_panel = trade_panel %>%
  select(year, everything())
