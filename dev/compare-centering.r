library(microbenchmark)
library(dplyr)
library(tidyr)
library(janitor)
library(purrr)

Sys.setenv(CAPYBARA_OPTIMIZATIONS = "yes")

# available cores - 1
# n_cores <- parallel::detectCores() - 1L

# n_cores <- parallel::detectCores() / 2

# n_cores <- c(2,4,6)

n_cores <- 4L

Sys.setenv(CAPYBARA_NCORES = n)

devtools::install(
  "~/Documents/capybara",
  upgrade = "never",
  dependencies = TRUE
)

y <- 2017L

data_bench <- readRDS(paste0("~/Documents/phd-thesis/capybara-benchmarks/tails-of-gravity-", y, ".rds"))

ap <- function() {
  fit <- capybara::fepoisson(
    trade ~ log(dist) + contig + fta | etfe + itfe | pair,
    data = data_bench
  )

  round(coef(fit), 4)
}

fp <- function() {
  fit <- capybara::fepoisson(
    trade ~ log(dist) + contig + fta | etfe + itfe | pair,
    data = data_bench,
    control = list(centering = "berge")
  )

  round(coef(fit), 4)
}

ap()
fp()

microbenchmark(
    ap = ap(),
    fp = fp(),
    times = 100L
)

# Unit: milliseconds
#  expr      min       lq     mean   median       uq      max neval cld
#    ap 272.8049 316.1880 330.4259 327.6882 342.1309 385.4831   100  a 
#    fp 214.0196 247.1539 263.2189 259.7161 272.6704 414.3084   100   b