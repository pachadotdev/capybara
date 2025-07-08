# Simple test
devtools::load_all()

# Base R
base_mod <- glm(mpg ~ wt + as.factor(cyl), mtcars, family = quasipoisson(link = "log"))
print("Base R:")
print(coef(base_mod))

# Your implementation  
your_mod <- fepoisson(mpg ~ wt | cyl | am, mtcars)
print("Your implementation:")
print(coef(your_mod))

# Fixed effects
print("Fixed effects:")
print(fixed_effects(your_mod))
