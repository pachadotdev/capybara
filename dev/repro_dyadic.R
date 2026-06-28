library(capybara)

set.seed(42)

# Simulate dyadic trade-like data where the structural regressors
# (ldist, lang, contig) are CONSTANT within an exporter-importer pair,
# i.e. perfectly absorbed by the `pair` fixed effect.
n_ctry <- 30
ctry <- sprintf("C%02d", seq_len(n_ctry))

grid <- expand.grid(exp = ctry, imp = ctry, stringsAsFactors = FALSE)
grid <- grid[grid$exp != grid$imp, ]

# pair id (undirected) and pair-level covariates
grid$pair <- paste0(pmin(grid$exp, grid$imp), "-", pmax(grid$exp, grid$imp))

pairs <- unique(grid$pair)
pair_dist <- setNames(runif(length(pairs), 1, 10), pairs)
pair_lang <- setNames(rbinom(length(pairs), 1, 0.3), pairs)
pair_contig <- setNames(rbinom(length(pairs), 1, 0.2), pairs)

grid$ldist <- log(pair_dist[grid$pair])
grid$lang <- pair_lang[grid$pair]
grid$contig <- pair_contig[grid$pair]

# response
mu <- exp(0.5 + 0.3 * grid$ldist - 0.2 * grid$lang + 0.1 * grid$contig)
grid$markup <- rpois(nrow(grid), mu) + 1
grid$weight <- runif(nrow(grid), 0.1, 1)

grid$reporter_code <- grid$exp
grid$partner_code <- grid$imp

cat("nobs:", nrow(grid), "\n")

fit1 <- fepoisson(
  markup ~ ldist + lang + contig | reporter_code + partner_code + pair,
  grid,
  weights = ~weight
)

print(summary(fit1))
cat("\nRaw coefficients:\n")
print(coef(fit1))
