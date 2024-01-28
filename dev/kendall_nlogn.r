devtools::clean_dll()
cpp11::cpp_register()
load_all()

fit <- feglm(
  trade ~ dist + lang + cntg + clny | exp_year + imp_year,
  trade_panel,
  family = poisson(link = "log")
)

summary(fit)

x <- trade_panel$trade
y <- predict(fit, type = "response")

pairwise_cor_(x, y)
pcaPP::cor.fk(x, y)
cor(x, y, method = "kendall")

length(x)

bench::mark(
  pairwise_cor_(x, y)
)
