load_all()

set.seed(200100)
d <- data.frame(
  y = rnorm(100),
  x1 = rnorm(100),
  f = factor(sample(1:2, 100, replace = TRUE))
)

d$x2 <- 2 * d$x1
fit1 <- lm(y ~ x1 + x2 + as.factor(f), data = d)
fit2 <- felm(y ~ x1 + x2 | f, data = d)