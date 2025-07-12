load_all()

d1 <- mtcars[mtcars$mpg >= quantile(mtcars$mpg, 0.25) & mtcars$mpg <= quantile(mtcars$mpg, 0.75), ]
d2 <- mtcars[mtcars$mpg < quantile(mtcars$mpg, 0.25) | mtcars$mpg > quantile(mtcars$mpg, 0.75), ]

# Linear model ----
mod_capybara <- felm(mpg ~ wt + disp | cyl, mtcars)
mod_fixest <- fixest::feols(mpg ~ wt + disp | cyl, mtcars)

pred1_capybara <- predict(mod_capybara, newdata = d1)
pred1_fixest <- predict(mod_fixest, newdata = d1)

pred2_capybara <- predict(mod_capybara, newdata = d2)
pred2_fixest <- predict(mod_fixest, newdata = d2)

expect_equal(pred1_capybara, pred1_fixest, tolerance = 1e-2)
expect_equal(pred2_capybara, pred2_fixest, tolerance = 1e-2)
