library(capybara)

mod <- feglm(
  am ~ wt + mpg | cyl,
  mtcars,
  family = binomial()
)

fe <- unname(drop(fixed_effects(mod)$cyl))
