if (!require("devtools")) { install.packages("devtools", repos = "https://cloud.r-project.org") }

Sys.setenv(CAPYBARA_DEBUG="no")

devtools::install(upgrade = "never", dependencies = FALSE, reload = TRUE)

# Data ----


url <- "https://data.mendeley.com/public-api/zip/n67gft8fvm/download/1"
zip <- "dev/tails-of-gravity.zip"
dir <- "dev/tails-of-gravity"

if (!file.exists(zip)) {
    download.file(url, destfile = zip, mode = "wb")
}

if (!file.exists(dir)) {
    unzip(zip, exdir = "dev/")
    zip2 <- "dev/The Tails of Gravity/Tails of Gravity data.zip"
    unzip(
        zip2,
        exdir = "dev/The Tails of Gravity"
    )

    unlink(zip2)

    file.rename("dev/The Tails of Gravity", dir)
    
    fout <- list.files(dir, pattern = "dta", full.names = TRUE)
    file.rename(
        list.files(dir, pattern = "dta", full.names = TRUE),
        tolower(gsub(" ", "-", fout))
    )
}

finp <- list.files(dir, pattern = "dta", full.names = TRUE)

# For OLS, with one clustering variable the sandwich estimator and M-estimator should be the same ----

bench_data <- haven::read_dta(finp)

bench_data$etfe <- paste0(bench_data$exporter, "_", bench_data$year)
bench_data$itfe <- paste0(bench_data$importer, "_", bench_data$year)
bench_data$pair <- paste0(bench_data$exporter, "_", bench_data$importer)

sandwich <- capybara::felm(
        log(trade_x) ~ log(dist) + contig + fta | etfe + itfe | pair,
        data = bench_data
    )

mestimator <- capybara::felm(
        log(trade_x) ~ log(dist) + contig + fta | etfe + itfe | pair,
        data = bench_data,
        control = list(vcov_type = "m-estimator")
    )

sandwich

mestimator

all.equal(vcov(sandwich), vcov(mestimator))

# Dyadic clustering adds correlations that make the previous equality not hold ----

# Dyadic clustering by exporter and importer
# Use the formula syntax: y ~ x | fe | entity1 + entity2
dyadic <- capybara::felm(
        log(trade_x) ~ log(dist) + contig + fta | etfe + itfe | exporter + importer,
        data = bench_data,
        control = list(vcov_type = "m-estimator-dyadic")
    )

sandwich

dyadic

vcov(sandwich)

vcov(dyadic)

all.equal(vcov(sandwich), vcov(dyadic))
