# Test with keep_mx = TRUE
devtools::load_all()

# Base R
base_mod <- glm(mpg ~ wt + as.factor(cyl), mtcars, family = quasipoisson(link = "log"))
print("Base R:")
print(coef(base_mod))

# Your implementation with keep_mx = TRUE  
your_mod <- fepoisson(mpg ~ wt | cyl, mtcars, control = fit_control(keep_mx = TRUE))
print("Your implementation:")
print(coef(your_mod))

# Check if MX is stored
print("MX stored?")
print("MX" %in% names(your_mod))

# Fixed effects
print("Fixed effects:")
print(fixed_effects(your_mod))
