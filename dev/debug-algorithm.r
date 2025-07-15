#!/usr/bin/env Rscript

# Debug our GLM algorithm step by step
library(capybara)

# Simple binomial test case from the failing tests
data(mtcars)

# Test multiple families
families_to_test <- list(
  list(name = "Binomial", formula_base = am ~ wt + mpg + as.factor(cyl), 
       formula_ours = am ~ wt + mpg | cyl, family = binomial()),
  list(name = "Gamma", formula_base = mpg ~ wt + am + as.factor(cyl), 
       formula_ours = mpg ~ wt + am | cyl, family = Gamma()),
  list(name = "Inverse Gaussian", formula_base = mpg ~ wt + am + as.factor(cyl), 
       formula_ours = mpg ~ wt + am | cyl, family = inverse.gaussian())
)

for (test_case in families_to_test) {
  cat("=== TESTING", test_case$name, "===\n")
  
  # Base R reference
  mod_base <- glm(test_case$formula_base, mtcars, family = test_case$family)
  cat("Base coefficients:\n")
  print(coef(mod_base))
  cat("Base deviance:", deviance(mod_base), "\n")
  
  # Our implementation
  mod_ours <- feglm(test_case$formula_ours, mtcars, family = test_case$family)
  cat("Our coefficients:\n")
  print(coef(mod_ours))
  cat("Our deviance:", mod_ours$deviance, "\n")
  
  # Comparison
  cat("Coefficient differences (should be ~0):\n")
  if (test_case$name == "Binomial") {
    print(coef(mod_ours) - coef(mod_base)[2:3])
  } else {
    print(coef(mod_ours) - coef(mod_base)[2:3])
  }
  cat("\n")
}
