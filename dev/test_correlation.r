load_all()

# fit <- fepoisson(mpg ~ wt | am, data = mtcars)

# y <- mtcars$mpg
# yhat <- predict(fit, type = "response")

# cor(y, yhat, method = "kendall")
# pairwise_cor_(y, yhat)

# summary(fit)

# fit2 <- glm(mpg ~ wt, data = mtcars, family = poisson(link = "log"))

# actual <- as.numeric(mtcars$mpg)
# predicted <- as.numeric(predict(fit2, type = "response"))
# r2 <- (stats::cor(actual, predicted, method = "kendall"))^2 # kendall mimics stata

library(tradepolicy)
library(dplyr)

ch1_application1 <- agtpa_applications %>%
  janitor::clean_names() %>%
  select(exporter, importer, pair_id, year, trade, dist, cntg, lang, clny) %>%
  filter(year %in% seq(1986, 2006, 4))

ch1_application1 <- ch1_application1 %>%
  mutate(
    log_trade = log(trade),
    log_dist = log(dist)
  )

ch1_application1 <- ch1_application1 %>%
  # Create Yit
  group_by(exporter, year) %>%
  mutate(
    y = sum(trade),
    log_y = log(y)
  ) %>%
  # Create Eit
  group_by(importer, year) %>%
  mutate(
    e = sum(trade),
    log_e = log(e)
  )

ch1_application1 <- ch1_application1 %>%
  # This merges the columns exporter/importer with year
  mutate(
    exp_year = paste0(exporter, year),
    imp_year = paste0(importer, year)
  )

ch1_application1 <- ch1_application1 %>%
  filter(exporter != importer)

fit_ppml <- fepoisson(trade ~ log_dist + cntg + lang + clny | exp_year + imp_year,
  data = ch1_application1
)

summary(fit_ppml)

y <- unlist(fit_ppml$data[, 1], use.names = FALSE)
yhat <- predict(fit_ppml, type = "link")

bench::mark(
  round(pairwise_cor_(y, yhat), 2),
  round(cor(y, yhat, method = "kendall"), 2)
)

# with O(n^2) pairwise correlation
# # A tibble: 2 × 13
#   expression                                      min median `itr/sec` mem_alloc
#   <bch:expr>                                 <bch:tm> <bch:>     <dbl> <bch:byt>
# 1 round(cor(y, yhat, method = "kendall"), 1)    6.32s  6.32s    0.158     2.15MB
# 2 round(pairwise_cor_(y, yhat), 1)             11.76s 11.76s    0.0851        0B

# using brain O(n log(n))
# # A tibble: 2 × 13
#   expression                                      min median `itr/sec` mem_alloc
#   <bch:expr>                                 <bch:tm> <bch:>     <dbl> <bch:byt>
# 1 round(pairwise_cor_(y, yhat), 2)             18.8ms   19ms    52.6          0B
# 2 round(cor(y, yhat, method = "kendall"), 2)     5.9s   5.9s     0.169    2.15MB
