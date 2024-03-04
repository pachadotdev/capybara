load_all()

# d <- trade_panel
# d$trade_100 <- ifelse(d$trade > 100, 1L, 0L)

# simulate a data frame with 1,000,000 rows with:
# trade_100: 0/1
# lang: 0/1
# clny: 0/1
# rta: 0/1
# year: 2000-2010
set.seed(200100)
d <- data.frame(
  trade_100 = sample(0:1, 1e6, replace = TRUE),
  lang = sample(0:1, 1e6, replace = TRUE),
  clny = sample(0:1, 1e6, replace = TRUE),
  rta = sample(0:1, 1e6, replace = TRUE),
  year = sample(2000:2010, 1e6, replace = TRUE)
)

unique(d$trade_100)
unique(d$lang)
unique(d$clny)
unique(d$rta)
unique(d$year)

# Fit 'feglm()'
load_all()
profvis::profvis(feglm(trade_100 ~ lang + clny + rta | year, d, family = binomial()))

# Compute average partial effects
# bench::mark(apes(mod))
