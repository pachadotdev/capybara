# Debug binomial specifically
load_all()

cat("=== BINOMIAL DEBUG ===\n")

# Base R reference
mod_base <- glm(am ~ wt + mpg + as.factor(cyl), mtcars, family = binomial())
cat("Base R coefficients:\n")
print(coef(mod_base))
cat("Base R deviance:", deviance(mod_base), "\n")

# fixest reference  
mod_fixest <- fixest::feglm(am ~ wt + mpg | cyl, mtcars, family = binomial())
cat("fixest coefficients:\n")
print(coef(mod_fixest))
cat("fixest deviance:", deviance(mod_fixest), "\n")

# Our implementation
mod_ours <- feglm(am ~ wt + mpg | cyl, mtcars, family = binomial())
cat("Our coefficients:\n")
print(coef(mod_ours))
cat("Our deviance:", mod_ours$deviance, "\n")

cat("\n=== COMPARISON ===\n")
cat("Difference from base R:\n")
print(coef(mod_ours) - coef(mod_base)[2:3])
cat("Difference from fixest:\n")  
print(coef(mod_ours) - coef(mod_fixest))

# Check starting values and convergence
cat("Our iterations:", mod_ours$iterations, "\n")
cat("Our converged:", mod_ours$converged, "\n")
