library(microbenchmark)

bench_data <- haven::read_dta("~/Documents/phd-thesis/capybara-benchmarks/The Tails of Gravity//Tails of Gravity data.dta")

bench_data$etfe <- paste0(bench_data$exporter, "_", bench_data$year)
bench_data$itfe <- paste0(bench_data$importer, "_", bench_data$year)
bench_data$pair <- paste0(bench_data$exporter, "_", bench_data$importer)

Sys.setenv(CAPYBARA_DEBUG="yes")

devtools::install(upgrade = "never", dependencies = FALSE, reload = TRUE)

capybara::fepoisson(
        trade_x ~ log(dist) + contig + fta | etfe + itfe | pair,
        data = bench_data
    )
