load_all()
  
mod_binom <- feglm(am ~ wt + mpg | cyl, mtcars, family = binomial())
mod_binom_fixest <- fixest::feglm(am ~ wt + mpg | cyl, mtcars, family = binomial())
coef(mod_binom)
coef(mod_binom_fixest)

fixed_effects(mod_binom)
fixest::fixef(mod_binom_fixest)
