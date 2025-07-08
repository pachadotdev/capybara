#!/usr/bin/env Rscript

# Verify fixed effects calculation
devtools::load_all()

# Base R approach
base_mod <- glm(mpg ~ wt + as.factor(cyl), mtcars, family = quasipoisson(link = "log"))
base_coef <- coef(base_mod)
cat("Base R coefficients:\n")
print(base_coef)

# Calculate what the fixed effects should be
# For cyl=4 (baseline): should be 0
# For cyl=6: should be base_coef["as.factor(cyl)6"] 
# For cyl=8: should be base_coef["as.factor(cyl)8"]

cat("\nExpected fixed effects (from base R):\n")
cat("cyl=4 (baseline): 0\n")
cat("cyl=6:", base_coef["as.factor(cyl)6"], "\n")
cat("cyl=8:", base_coef["as.factor(cyl)8"], "\n")

# Your implementation
your_mod <- fepoisson(mpg ~ wt | cyl | am, mtcars)
your_coef <- coef(your_mod)
your_fe <- fixed_effects(your_mod)

cat("\nYour implementation:\n")
cat("wt coefficient:", your_coef, "\n")
cat("Fixed effects:\n")
print(your_fe)

# Compare
cat("\nComparison:\n")
cat("wt coefficient difference:", your_coef - base_coef["wt"], "\n")
cat("Relative difference:", (your_coef - base_coef["wt"]) / base_coef["wt"] * 100, "%\n")
