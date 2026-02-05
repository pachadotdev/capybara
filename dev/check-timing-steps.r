library(microbenchmark)

bench_data <- readRDS("dev/bench_data.rds")

Sys.setenv(CAPYBARA_DEBUG="yes")

devtools::install(upgrade = "never", dependencies = FALSE, reload = TRUE)

capybara::fepoisson(
        trade ~ log(dist) + contig + fta | etfe + itfe | pair,
        data = bench_data,
        control = list(check_separation = T)
    )
