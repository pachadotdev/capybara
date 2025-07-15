load_all()

cat("=== DEBUGGING BINOMIAL GLM CONVERGENCE ===\n\n")

# Test case from failing test
cat("## Running test case:\n")
cat("capybara: mod_binom <- feglm(am ~ wt + mpg | cyl, mtcars, family = binomial())\n")
cat("base R:   mod_binom_base <- glm(am ~ wt + mpg + as.factor(cyl), mtcars, family = binomial())\n\n")

mod_binom <- feglm(am ~ wt + mpg | cyl, mtcars, family = binomial())
mod_binom_base <- glm(am ~ wt + mpg + as.factor(cyl), mtcars, family = binomial())

cat("## Results:\n")
cat("capybara coefficients:\n")
print(coef(mod_binom))
cat("\nbase R coefficients (relevant):\n")
print(coef(mod_binom_base)[2:3])

cat("\n## Coefficient comparison:\n")
coef_diff <- coef(mod_binom) - coef(mod_binom_base)[2:3]
cat("Differences (capybara - base R):\n")
print(coef_diff)
cat("Ratio (capybara / base R):\n")
print(coef(mod_binom) / coef(mod_binom_base)[2:3])

cat("\n## Summary info:\n")
cat("capybara converged:", mod_binom$converged, "\n")
cat("capybara iterations:", mod_binom$iter, "\n")
cat("base R converged:", mod_binom_base$converged, "\n")

# Try with very strict tolerances
cat("\n## Testing with strict tolerances:\n")
strict_ctrl <- feglm_control(dev_tol = 1e-10, center_tol = 1e-10, iter_center_max = 1000)
mod_binom_strict <- feglm(am ~ wt + mpg | cyl, mtcars, family = binomial(), control = strict_ctrl)
cat("strict capybara coefficients:\n")
print(coef(mod_binom_strict))
cat("strict converged:", mod_binom_strict$converged, "\n")
cat("strict iterations:", mod_binom_strict$iter, "\n")

# Let's also test a simpler case
cat("\n## Testing simpler case (no fixed effects):\n")
mod_simple <- feglm(am ~ wt + mpg, mtcars, family = binomial())
mod_simple_base <- glm(am ~ wt + mpg, mtcars, family = binomial())
cat("simple capybara coefficients:\n")
print(coef(mod_simple))
cat("simple base R coefficients:\n")
print(coef(mod_simple_base))
cat("simple differences:\n")
print(coef(mod_simple) - coef(mod_simple_base)[2:3])
