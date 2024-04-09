library(tidyverse)
# library(alpaca)
# library(capybara)
library(devtools)

load_all()

# trade <- capybara::trade_panel %>%
trade <- trade_panel %>%
  mutate(
    exporter = str_sub(exp_year, 1, 3),
    importer = str_sub(imp_year, 1, 3),
    pair_id_2 = ifelse(exporter == importer, "0-intra", pair),

    # Set reference country
    exporter = ifelse(exporter == "DEU", "0-DEU", exporter),
    importer = ifelse(importer == "DEU", "0-DEU", importer)
  ) %>%
  # Sort by importer
  arrange(importer) %>%
  # Compute sum of trade by pair
  group_by(pair) %>%
  mutate(sum_trade = sum(trade)) %>%
  ungroup()

# Poisson regression with Capybara works fine
# fit_capybara <- capybara::fepoisson(
object <- fepoisson(
  trade ~ rta | exp_year + imp_year + pair_id_2,
  data = trade %>% filter(sum_trade > 0)
)

foo <- fixed_effects(object)

class(foo)
class(foo$exp_year)

head(foo$exp_year)

summary(object)

# Error when using fixed_effects()
options(error = function() traceback(3))
foo <- fixed_effects(object)
bar <- alpaca::getFEs(object)

names(foo)
head(foo$exp_year)
head(bar$exp_year)

all.equal(foo$exp_year, bar$exp_year)
all.equal(foo$imp_year, bar$imp_year)
all.equal(foo$pair_id_2, bar$pair_id_2)

saveRDS(
  list(model = object, fes = foo),
  "dev/cass_bug.rds"
)
