load_all()

fit <- felm(mpg ~ wt | cyl, mtcars)

fit$coefficients
fit$r.squared
fit$adj.r.squared

fit2 <- lm(mpg ~ wt + as.factor(cyl), mtcars)
fit2$coefficients
summary(fit2)$r.squared
summary(fit2)$adj.r.squared
