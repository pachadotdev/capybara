devtools::load_all()

d1 <- mtcars[mtcars$mpg >= quantile(mtcars$mpg, 0.25) & mtcars$mpg <= quantile(mtcars$mpg, 0.75), ]
d2 <- mtcars[mtcars$mpg < quantile(mtcars$mpg, 0.25) | mtcars$mpg > quantile(mtcars$mpg, 0.75), ]

m1_binom <- feglm(am ~ wt + disp | cyl, mtcars, family = binomial())
m2_binom <- glm(am ~ wt + disp + as.factor(cyl), mtcars, family = binomial())

names(m1_binom)

m1_binom$coefficients
m1_binom$fixed.effects

pred1_binom <- predict(m1_binom, newdata = d1, type = "response")
pred2_binom <- predict(m1_binom, newdata = d2, type = "response")
