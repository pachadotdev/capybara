load_all()

cat("\n=== Fixest GLM ===\n")
mod_fixest <- fixest::feglm(am ~ wt + mpg | cyl, mtcars, family = binomial())
cat("Fixest coefficients:\n")
print(coef(mod_fixest))
cat("Fixest deviance:", mod_fixest$deviance, "\n")
cat("Fixest iterations:", mod_fixest$iterations, "\n")

cat("\n=== Capybara feglm (with debug output) ===\n")
mod_cap <- feglm(am ~ wt + mpg | cyl, mtcars, family = binomial())
cat("Capybara coefficients:\n")
print(coef(mod_cap))
cat("Capybara deviance:", mod_cap$deviance, "\n")
cat("Capybara iterations:", mod_cap$iter, "\n")

cat("\n=== Comparison ===\n")
fixest_coef <- coef(mod_fixest)[c("wt", "mpg")]
cap_coef <- coef(mod_cap)
cat("Coefficient differences:\n")
print(cap_coef - fixest_coef)
cat("Deviance difference:", mod_cap$deviance - mod_fixest$deviance, "\n")
