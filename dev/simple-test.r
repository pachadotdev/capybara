load_all()
  
capybara_pois <- fepoisson(mpg ~ wt + disp | cyl, mtcars)
fixest_pois <- fixest::fepois(mpg ~ wt + disp | cyl, mtcars)

coef(capybara_pois)
coef(fixest_pois)

fixed_effects(capybara_pois)
fixest::fixef(fixest_pois)
