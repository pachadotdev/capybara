# Debug script to understand base R's fixed effects identification
library(data.table)

# Same data as our tests
data(mtcars)
mtcars$cyl <- factor(mtcars$cyl)

# Fit with base R
model_r <- glm(vs ~ wt + cyl, data = mtcars, family = poisson())

# Extract key components
print("=== BASE R MODEL ===")
print(summary(model_r))
print("Coefficients:")
print(coef(model_r))

# The key insight: base R includes an intercept + factor levels
# Let's see how this works exactly
print("\n=== MODEL MATRIX ===")
X <- model.matrix(model_r)
print("First few rows of design matrix:")
print(head(X))
print("Unique values in design matrix:")
print(unique(X))

print("\n=== MANUAL VERIFICATION ===")
# Manual calculation of what base R does
y <- mtcars$vs
wt <- mtcars$wt
cyl <- mtcars$cyl

# Base R design matrix has intercept + dummy vars for cyl6 and cyl8
# This means:
# - cyl=4: intercept only (reference level)
# - cyl=6: intercept + cyl6 dummy  
# - cyl=8: intercept + cyl8 dummy

# So the fixed effects should be:
# - For cyl=4: intercept = 3.6897484
# - For cyl=6: intercept + cyl6 = 3.6897484 + (-0.1480081) = 3.5417403
# - For cyl=8: intercept + cyl8 = 3.6897484 + (-0.2639083) = 3.4258401

intercept <- coef(model_r)["(Intercept)"]
cyl6_effect <- coef(model_r)["cyl6"] 
cyl8_effect <- coef(model_r)["cyl8"]

expected_fe <- c(
  intercept,                    # cyl=4 (reference)
  intercept + cyl6_effect,      # cyl=6
  intercept + cyl8_effect       # cyl=8
)

print("Expected fixed effects:")
print(expected_fe)

# Verify this matches our target values
target <- c(3.6897484, 3.5417403, 3.4258401)
print("Target fixed effects:")
print(target)
print("Difference:")
print(expected_fe - target)
