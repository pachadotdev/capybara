load_all()

library(tradepolicy)
library(dplyr)
library(tidyr)

ch1_application2 <- agtpa_applications %>%
  janitor::clean_names() %>%
  select(exporter, importer, pair_id, year, trade, dist, cntg, lang, clny) %>%
  # this filter covers both OLS and PPML
  filter(year %in% seq(1986, 2006, 4)) %>%
  mutate(
    # variables for both OLS and PPML
    exp_year = paste0(exporter, year),
    imp_year = paste0(importer, year),
    year = paste0("log_dist_", year),
    log_trade = log(trade),
    log_dist = log(dist),
    smctry = ifelse(importer != exporter, 0, 1),

    # PPML specific variables
    log_dist_intra = log_dist * smctry,
    intra_pair = ifelse(exporter == importer, exporter, "inter")
  ) %>%
  pivot_wider(names_from = year, values_from = log_dist, values_fill = 0) %>%
  mutate(across(log_dist_1986:log_dist_2006, function(x) x * (1 - smctry)))

microbenchmark::microbenchmark(
  times = 1L,
  feglm(
    trade ~ 0 + log_dist_1986 + log_dist_1990 + log_dist_1994 +
      log_dist_1998 + log_dist_2002 + log_dist_2006 + cntg + lang + clny +
      log_dist_intra | exp_year + imp_year,
    data = ch1_application2,
    family = poisson(link = "log")
  )
)

microbenchmark::microbenchmark(
  times = 1L,
  glm(
    trade ~ 0 + log_dist_1986 + log_dist_1990 + log_dist_1994 +
      log_dist_1998 + log_dist_2002 + log_dist_2006 + cntg + lang + clny +
      log_dist_intra + as.factor(exp_year) + as.factor(imp_year),
    data = ch1_application2,
    family = poisson(link = "log")
  )
)
