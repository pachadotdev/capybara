sink("dev/check-appml.txt")

library(capybara)
library(data.table)
library(readstata13)
library(ggplot2)

tails <- read.dta13("dev/tails-of-gravity/Tails of Gravity data.dta")

setDT(tails)

required_vars <- c("iso_o", "iso_d", "year", "pairid", "trade_x", "gdp_o", "EIA",
                   "expyear2", "impyear2")

tails <- tails[, ..required_vars]

# tails preparation
tails[, INTER := as.integer(iso_o != iso_d)]
setorder(tails, year, pairid)
tails[, trade := trade_x]
tails[is.na(trade), trade := 0]
tails[, agg_exports := sum(trade), by = .(iso_o, year)]
tails[iso_o == iso_d, trade := gdp_o - agg_exports]
tails[is.na(trade) | trade < 0, trade := 0]
tails[, trade_all := trade / 1e6]
tails <- tails[year >= 1962]
tails[iso_o == iso_d, rta := 0]

# EIA is a factor with labels. Map to Stata numeric coding:
# Stata: 0=No Agreement, 1=OneWay, 2=TwoWay, 3=FTA, >=4=CUCMECU
# R factor levels: "No Country", "No Agreement" -> 0; others have agreements
tails[, `:=`(
  EIAp = as.integer(!EIA %in% c("No Country", "No Agreement") & !is.na(EIA)),
  OneWay = as.integer(EIA == "Non-Reciprocal PTA"),
  TwoWay = as.integer(EIA == "Preferential Trade Agreement"),
  FTA = as.integer(EIA == "Free Trade Agreement"),
  CUCMECU = as.integer(EIA %in% c("Customs Union", "Common Market", "Economic Union"))
)]

# Remove rows with missing EIA values
tails <- tails[!is.na(EIAp)]

# Create interaction term for INTER and year
tails[, INTER_YEAR := interaction(INTER, year, sep = "_")]

# Expectile 50% ----

fepoisson(trade_all ~ EIAp | INTER_YEAR + expyear2 + impyear2 + pairid, data = tails)

fepoisson_asymmetric(trade_all ~ EIAp | INTER_YEAR + expyear2 + impyear2 + pairid, data = tails,
  control = fit_control(expectile = 0.5, expectile_trace = TRUE, check_separation = TRUE))

fepoisson(trade_all ~ EIAp | INTER_YEAR + expyear2 + impyear2 + pairid, data = tails,
  control = fit_control(expectile = 0.5, expectile_trace = TRUE, check_separation = FALSE))

# Expectile 10% ----

fepoisson_asymmetric(trade_all ~ EIAp | INTER_YEAR + expyear2 + impyear2 + pairid, data = tails,
  control = fit_control(expectile = 0.1, expectile_trace = TRUE, check_separation = TRUE))

fepoisson_asymmetric(trade_all ~ EIAp | INTER_YEAR + expyear2 + impyear2 + pairid, data = tails,
  control = fit_control(expectile = 0.1, expectile_trace = TRUE, check_separation = FALSE))

# Expectile 90% ----

fepoisson_asymmetric(trade_all ~ EIAp | INTER_YEAR + expyear2 + impyear2 + pairid, data = tails,
  control = fit_control(expectile = 0.9, expectile_trace = TRUE, check_separation = TRUE))

fepoisson_asymmetric(trade_all ~ EIAp | INTER_YEAR + expyear2 + impyear2 + pairid, data = tails,
  control = fit_control(expectile = 0.9, expectile_trace = TRUE, check_separation = FALSE))

sink()
