load_all()

d1 <- mtcars[mtcars$mpg >= quantile(mtcars$mpg, 0.25) & mtcars$mpg <= quantile(mtcars$mpg, 0.75), ]
d2 <- mtcars[mtcars$mpg < quantile(mtcars$mpg, 0.25) | mtcars$mpg > quantile(mtcars$mpg, 0.75), ]

m1_capybara <- feglm(am ~ wt + disp | cyl, mtcars, family = binomial())
m1_fixest <- fixest::feglm(am ~ wt + disp | cyl, mtcars, family = binomial())

pred1_capybara <- predict(m1_capybara, newdata = d1, type = "response")
pred1_fixest <- predict(m1_fixest, newdata = d1, type = "response")

# Compare with base R Binomial
expect_equal(pred1_capybara, pred1_fixest, tolerance = 1e-2)
