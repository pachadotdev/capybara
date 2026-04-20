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

# fepoisson(trade_all ~ EIAp | INTER_YEAR + expyear2 + impyear2 + pairid, data = tails)

# Separation detected: 268578 observation(s) with perfect prediction were excluded from estimation.
# Formula: trade_all ~ EIAp | INTER_YEAR + expyear2 + impyear2 + pairid

# Family: Poisson

# Estimates:

# |      | Estimate | Std. Error | z value  | Pr(>|z|)  |
# |------|----------|------------|----------|-----------|
# | EIAp |   0.1975 |     0.0003 | 664.3335 | 0.0000 ** |

# Significance codes: ** p < 0.01; * p < 0.05; + p < 0.10

# Pseudo R-squared: 1.349e-10 

# Number of observations: Full 1769921; Separated 268578; Perfect classification 0 

# Number of Fisher Scoring iterations: 16 

# fepoisson_asymmetric(trade_all ~ EIAp | INTER_YEAR + expyear2 + impyear2 + pairid, data = tails,
#   control = fit_control(expectile = 0.5, expectile_trace = T))

# Separation detected: 268578 observation(s) excluded
# APPML: expectile = 0.5, using standard Poisson (no iteration)
# % negative residuals = 77.924%
# Formula: trade_all ~ EIAp | INTER_YEAR + expyear2 + impyear2 + pairid

# Family: Poisson

# Estimates:

# |      | Estimate | Std. Error | z value  | Pr(>|z|)  |
# |------|----------|------------|----------|-----------|
# | EIAp |   0.1975 |     0.0003 | 664.3335 | 0.0000 ** |

# Significance codes: ** p < 0.01; * p < 0.05; + p < 0.10

# Number of observations: Full 1501343; Separated 268578; Perfect classification 0 

# Number of Fisher Scoring iterations: 16 

# Expectile 10% ----

# fepoisson_asymmetric(trade_all ~ EIAp | INTER_YEAR + expyear2 + impyear2 + pairid, data = tails,
#   control = fit_control(expectile = 0.1, expectile_trace = T, check_separation = T))

# Separation detected: 268578 observation(s) excluded

# Iteration 1: objective function = 7.243773e-03
# Iteration 2: objective function = 8.270714e-04
# Iteration 3: objective function = 6.484599e-06
# Iteration 4: objective function = 4.471886e-09
# Iteration 5: objective function = 0.000000e+00

# APPML converged after 5 iterations
# Tolerance = 1.00e-12, Objective = 0.000000e+00
# % negative residuals = 64.475%
# Expectile = 0.100
# Formula: trade_all ~ EIAp | INTER_YEAR + expyear2 + impyear2 + pairid

# Family: Poisson

# Estimates:

# |      | Estimate | Std. Error | z value  | Pr(>|z|)  |
# |------|----------|------------|----------|-----------|
# | EIAp |   0.3140 |     0.0007 | 424.0580 | 0.0000 ** |

# Significance codes: ** p < 0.01; * p < 0.05; + p < 0.10

# Number of observations: Full 1501343; Separated 268578; Perfect classification 0 

# Number of Fisher Scoring iterations: 1

# fepoisson_asymmetric(trade_all ~ EIAp | INTER_YEAR + expyear2 + impyear2 + pairid, data = tails,
#   control = fit_control(expectile = 0.1, expectile_trace = T, check_separation = F))

# > fepoisson_asymmetric(trade_all ~ EIAp | INTER_YEAR + expyear2 + impyear2 + pairid, data = tails,
# +   control = fit_control(expectile = 0.1, expectile_trace = T, check_separation = F))

# Iteration 1: objective function = 7.242208e-03
# Iteration 2: objective function = 8.276116e-04
# Iteration 3: objective function = 6.474723e-06
# Iteration 4: objective function = 4.721983e-09
# Iteration 5: objective function = 0.000000e+00

# APPML converged after 5 iterations
# Tolerance = 1.00e-12, Objective = 0.000000e+00
# % negative residuals = 69.866%
# Expectile = 0.100
# Formula: trade_all ~ EIAp | INTER_YEAR + expyear2 + impyear2 + pairid

# Family: Poisson

# Estimates:

# |      | Estimate | Std. Error | z value  | Pr(>|z|)  |
# |------|----------|------------|----------|-----------|
# | EIAp |   0.3140 |     0.0007 | 424.0579 | 0.0000 ** |

# Significance codes: ** p < 0.01; * p < 0.05; + p < 0.10

# Number of observations: Full 1769921; Missing 0; Perfect classification 0 

# Number of Fisher Scoring iterations: 1 

# Expectile 90% ----

fepoisson_asymmetric(trade_all ~ EIAp | INTER_YEAR + expyear2 + impyear2 + pairid, data = tails,
  control = fit_control(expectile = 0.9, expectile_trace = T, check_separation = T))

# Separation detected: 268578 observation(s) excluded

# Iteration 1: objective function = 3.748429e-03
# Iteration 2: objective function = 2.965636e-05
# Iteration 3: objective function = 4.612421e-09
# Iteration 4: objective function = 3.240890e-12
# Iteration 5: objective function = 0.000000e+00

# APPML converged after 5 iterations
# Tolerance = 1.00e-12, Objective = 0.000000e+00
# % negative residuals = 89.098%
# Expectile = 0.900
# Formula: trade_all ~ EIAp | INTER_YEAR + expyear2 + impyear2 + pairid

# Family: Poisson

# Estimates:

# |      | Estimate | Std. Error | z value  | Pr(>|z|)  |
# |------|----------|------------|----------|-----------|
# | EIAp |   0.1309 |     0.0006 | 234.6403 | 0.0000 ** |

# Significance codes: ** p < 0.01; * p < 0.05; + p < 0.10

# Number of observations: Full 1501343; Separated 268578; Perfect classification 0 

# Number of Fisher Scoring iterations: 1 

fepoisson_asymmetric(trade_all ~ EIAp | INTER_YEAR + expyear2 + impyear2 + pairid, data = tails,
  control = fit_control(expectile = 0.9, expectile_trace = T, check_separation = F))

# Iteration 1: objective function = 3.749316e-03
# Iteration 2: objective function = 2.958251e-05
# Iteration 3: objective function = 4.523513e-09
# Iteration 4: objective function = 5.860800e-12
# Iteration 5: objective function = 0.000000e+00

# APPML converged after 5 iterations
# Tolerance = 1.00e-12, Objective = 0.000000e+00
# % negative residuals = 90.753%
# Expectile = 0.900
# Formula: trade_all ~ EIAp | INTER_YEAR + expyear2 + impyear2 + pairid

# Family: Poisson

# Estimates:

# |      | Estimate | Std. Error | z value  | Pr(>|z|)  |
# |------|----------|------------|----------|-----------|
# | EIAp |   0.1309 |     0.0006 | 234.6482 | 0.0000 ** |

# Significance codes: ** p < 0.01; * p < 0.05; + p < 0.10

# Number of observations: Full 1769921; Missing 0; Perfect classification 0 

# Number of Fisher Scoring iterations: 1
