library(capybara)

fit1 <- fepoisson(
  y ~ x1 + x2 + x3 + x4,
  data = correia2019$example1
)

fit2 <- fepoisson(
  y ~ x1 + x2 + x3 + x4,
  data = correia2019$example1,
  control = fit_control(check_separation = FALSE)
)

summary_table(fit1, fit2)

round(cor(predict(fit1)[-5], correia2019$example1$y[-5])^2, 3)

round(cor(predict(fit2), correia2019$example1$y)^2, 3)
