load_all()

mod <- feglm(
  formula = trade ~ dist + lang + cntg + clny | exp_year + imp_year,
  data = trade_panel,
  family = poisson(link = "log")
)

# mod <- felm(
#   formula = trade ~ dist + lang + cntg + clny | exp_year + imp_year,
#   data = trade_panel
# )

summary(mod)
