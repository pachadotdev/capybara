devtools::load_all()
mod3 <- felm(mpg ~ wt | cyl + am, mtcars)
mod3$coefficients
mod3$fixed.effects

mod3_fixest <- fixest::feols(mpg ~ wt | cyl + am, mtcars)
mod3_fixest$coefficients
fixest::fixef(mod3_fixest)

lm(mpg ~ wt + as.factor(cyl) + as.factor(am), mtcars)
