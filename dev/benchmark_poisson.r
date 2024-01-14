load("data/trade_panel.rda")

library(magrittr)

fml <- trade ~ dist + lang + cntg + clny | exp_year + imp_year

coef_capybara <- function(fml, data) {
  capybara::feglm(fml, data = data, family = poisson(link = "log")) %>%
    coef() %>%
    as.numeric() %>%
    unname() %>%
    round(1)
}

# coef_capybara(fml, trade_panel)

coef_alpaca <- function(fml, data) {
  alpaca::feglm(fml, data = data, family = poisson(link = "log")) %>%
    coef() %>%
    as.numeric() %>%
    unname() %>%
    round(1)
}

# coef_alpaca(fml, trade_panel)

coef_fixest <- function(fml, data) {
  fixest::feglm(fml, data = data, family = poisson(link = "log")) %>%
    coef() %>%
    as.numeric() %>%
    unname() %>%
    round(1)
}

# coef_fixest(fml, trade_panel)

# round(coef_capybara(fml, trade_panel), 1)
# round(coef_alpaca(fml, trade_panel), 1)
# round(coef_fixest(fml, trade_panel), 1)

bench::mark(
  iterations = 100L,
  capybara = coef_capybara(fml, trade_panel),
  alpaca = coef_alpaca(fml, trade_panel),
  fixest = coef_fixest(fml, trade_panel)
)

# # A tibble: 3 × 13
#   expression      min median `itr/sec` mem_alloc `gc/sec` n_itr  n_gc total_time
#   <bch:expr> <bch:tm> <bch:>     <dbl> <bch:byt>    <dbl> <int> <dbl>   <bch:tm>
# 1 capybara      124ms  146ms      6.34     264MB    55.3    100   872      15.8s
# 2 alpaca        749ms  826ms      1.19     288MB    13.6    100  1139       1.4m
# 3 fixest        164ms  178ms      5.43      49MB     8.47   100   156      18.4s
# # ℹ 4 more variables: result <list>, memory <list>, time <list>, gc <list>
