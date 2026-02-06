Sys.setenv(CAPYBARA_DEBUG="no")

# clean_dll()

# devtools::check()

devtools::install(upgrade = "never", dependencies = FALSE, reload = TRUE)

library(microbenchmark)

# library(dplyr)
# library(tidyr)
# library(janitor)

# y = 2017

# bench_data <- haven::read_dta(
#     "~/Documents/phd-thesis/capybara-benchmarks/The Tails of Gravity/Tails of Gravity data.dta"
# ) %>%
#     clean_names() %>%
#     select(year, iso_o, iso_d, dist, contig, fta, trade = trade_x) %>%
#     mutate(
#         etfe = paste0(iso_o, year),
#         itfe = paste0(iso_d, year),
#         pair = paste0(iso_o, "_", iso_d)
#     ) %>%
#     select(-c(iso_o, iso_d)) %>%
#     drop_na() %>%
#     filter(
#         year %in% seq(1967, 2017, by = 10),
#         etfe != itfe
#     ) %>%
#     filter(year <= y)

# saveRDS(bench_data, file = "dev/bench_data.rds", compress = "xz")

bench_data <- readRDS("dev/bench_data.rds")

dim(bench_data)
# [1] 88728     8

length(unique(c(bench_data$etfe, bench_data$itfe)))
# [1] 1024

bench_capybara <- function(check_separation = FALSE) {
    fit <- capybara::fepoisson(
        trade ~ log(dist) + contig + fta | etfe + itfe | pair,
        data = bench_data,
        control = list(check_separation = T)
    )

    round(coef(fit), 4)
}

bench_fixest <- function() {
    fit <- fixest::fepois(
        trade ~ log(dist) + contig + fta | etfe + itfe,
        data = bench_data,
        cluster = ~pair
    )

    # add these for a fair comparison
    fit$fixed_effects <- fixest::fixef(fit)

    round(coef(fit), 4)
}

# bench_capybara(check_separation = FALSE)

bench_capybara(check_separation = TRUE)

bench_fixest()

bench <- suppressMessages({
    summary(microbenchmark(
        capybara_check = bench_capybara(check_separation = FALSE),
        capybara_nocheck = bench_capybara(check_separation = TRUE),
        times = 30L,
        unit = "s"
    ))
})

bench
