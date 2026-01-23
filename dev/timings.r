library(bench)
library(dplyr)
library(tidyr)
library(janitor)
library(purrr)

# available cores - 1
n_cores <- parallel::detectCores() - 1L
# n_cores <- parallel::detectCores() / 2
Sys.setenv(CAPYBARA_OPTIMIZATIONS = "yes")
Sys.setenv(CAPYBARA_NCORES = n_cores)
devtools::install(
    "~/Documents/capybara",
    upgrade = "never",
    dependencies = TRUE
)

# dataset ----

url <- "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/n67gft8fvm-1.zip"
zip <- "~/Documents/phd-thesis/capybara-benchmarks/The Tails of Gravity.zip"
dir <- "~/Documents/phd-thesis/capybara-benchmarks/The Tails of Gravity"

if (!file.exists(zip)) {
    download.file(url, destfile = zip, mode = "wb")
}

if (!file.exists(dir)) {
    unzip(zip, exdir = "~/Documents/phd-thesis/capybara-benchmarks/")
    zip2 <- "~/Documents/phd-thesis/capybara-benchmarks/The Tails of Gravity/Tails of Gravity data.zip"
    unzip(
        zip2,
        exdir = "~/Documents/phd-thesis/capybara-benchmarks/The Tails of Gravity"
    )
}

data_table4 <- haven::read_dta(
    "~/Documents/phd-thesis/capybara-benchmarks/The Tails of Gravity/Tails of Gravity data.dta"
)

glimpse(data_table4)

min(data_table4$year)
max(data_table4$year)
seq(min(data_table4$year), max(data_table4$year), 5)

data_table4 <- data_table4 %>%
    clean_names() %>%
    select(
        year,
        iso_o,
        iso_d,
        dist,
        contig,
        fta,
        trade = trade_x
    ) %>%
    mutate(
        etfe = paste0(iso_o, year),
        itfe = paste0(iso_d, year),
        pair = paste0(iso_o, "_", iso_o)
    ) %>%
    filter(year %in% seq(min(data_table4$year), max(data_table4$year), 5)) %>%
    select(-c(year, iso_o, iso_d))

data_table4 <- data_table4 %>%
    filter(etfe != itfe) %>%
    drop_na()

glimpse(data_table4)

bench_combined <- map_df(
    # seq(0.25, 1, 0.25),
    1,
    function(s) {
        set.seed(1234)

        data_sample <- data_table4 %>%
            sample_frac(s)

        bench_capybara <- function(check_separation = FALSE) {
            fit <- capybara::fepoisson(
                trade ~ log(dist) + contig + fta | etfe + itfe | pair,
                data = data_sample,
                control = list(check_separation = check_separation)
            )

            round(coef(fit), 4)
        }

        bench_fixest <- function() {
            fit <- fixest::feglm(
                trade ~ log(dist) + contig + fta | etfe + itfe,
                data = data_sample,
                cluster = ~pair,
                family = "poisson"
            )

            # add fixed effects for a fair comparison
            fit$fixed_effects <- fixest::fixef(fit)

            round(coef(fit), 4)
        }

        bench_results <- mark(
            capybara_no_check = bench_capybara(check_separation = FALSE),
            fixest = bench_fixest(),
            iterations = 100L,
            memory = F
        )

        bench_results <- bench_results %>%
            select(expression, time)

        bench_results %>%
            mutate(
                p0_time = map_dbl(bench_results$time, function(x) { quantile(x, 0.00) }),
                p25_time = map_dbl(bench_results$time, function(x) { quantile(x, 0.25) }),
                p50_time = map_dbl(bench_results$time, function(x) { quantile(x, 0.50) }),
                p75_time = map_dbl(bench_results$time, function(x) { quantile(x, 0.75) }),
                p100_time = map_dbl(bench_results$time, function(x) { quantile(x, 1.00) })
            ) %>%
            select(-time) %>%
            mutate(nobs = nrow(data_sample))
    }
)

bench_combined <- bench_combined %>%
    mutate(expression = as.character(expression)) %>%
    select(nobs, expression, p0_time:p100_time)

bench_combined_path <- paste0("dev/timings_", gsub("\\s+|:", "_", Sys.time()), ".rds")

saveRDS(bench_combined, bench_combined_path, compress = "xz")

finp <- list.files(
    "dev/",
    pattern = "timings_.*\\.rds$",
    full.names = TRUE
)

finp_latest <- finp[which.max(file.info(finp)$ctime)]
finp_2nd_latest <- finp[order(file.info(finp)$ctime, decreasing = TRUE)[2]]

if (is.na(finp_2nd_latest)) {
    finp_2nd_latest <- finp_latest
}

bench_new <- readRDS(finp_latest) %>%
    select(nobs, expression, p50_time) %>%
    rename(p50_time_new = p50_time)

bench_old <- readRDS(finp_2nd_latest) %>%
    select(nobs, expression, p50_time) %>%
    rename(p50_time_old = p50_time) %>%
    left_join(bench_new, by = c("nobs", "expression")) %>%
    mutate(p50_change = (p50_time_new - p50_time_old) / p50_time_old * 100)

bench_old

readr::write_csv(
    bench_old,
    paste0("dev/timings_comparison_", gsub("\\s+|:", "_", Sys.time()), ".csv")
)
