devtools::install(".", upgrade = "never")

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

# Expected result ----

# From the Tails of Gravity:

# Table 4  Baseline results for selected expectiles.
#        10th     50th     90th     10th–90th
# EIAijt 0.314∗∗∗ 0.198∗∗∗ 0.131∗∗∗ 0.183∗∗∗
#        (0.057)  (0.042)  (0.032)  (0.048)
# Note: All models include Exporter-year, Importer-year, and Pair fixed effects.
# Number of observations is 1,499,735. Estimates for the 50th expectile correspond to the standard
# PPML estimates for the mean. Clustered standard errors by country-pair are in parentheses,
# ∗ p < .10, ∗∗ p < .05, ∗∗∗ p < .01.

# Expectile 50% ----

fit1 <- fepoisson(trade_all ~ EIAp | INTER_YEAR + expyear2 + impyear2 + pairid, data = tails,
  control = fit_control(return_fe = TRUE, check_separation = TRUE))

fit2 <- fepoisson(trade_all ~ EIAp | INTER_YEAR + expyear2 + impyear2 + pairid, data = tails,
  control = fit_control(return_fe = TRUE, check_separation = FALSE))

fit1$conv
fit2$conv

fit1$nms_fe
fit1$fe_levels
fit1$separated_obs
fit1$fixed_effects

length(fit1$fixed_effects)

for (i in seq_along(fit1$fixed_effects)) {
  cat("FE", i, "levels:", length(fit1$fixed_effects[[i]]), "\n")
}

length(fit2$fixed_effects)

for (i in seq_along(fit2$fixed_effects)) {
  cat("FE", i, "levels:", length(fit2$fixed_effects[[i]]), "\n")
}

all.equal(fit1$fixed_effects, fit2$fixed_effects)

fit1
fit2

fepoisson_asymmetric(trade_all ~ EIAp | INTER_YEAR + expyear2 + impyear2 + pairid, data = tails,
  control = fit_control(expectile = 0.5, expectile_trace = TRUE, check_separation = TRUE))

fepoisson_asymmetric(trade_all ~ EIAp | INTER_YEAR + expyear2 + impyear2 + pairid, data = tails,
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
