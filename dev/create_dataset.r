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

use_data(trade_panel, overwrite = T)
