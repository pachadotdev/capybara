test_that("vcov works", {
  m1 <- fepoisson(
    trade ~ log_dist + lang + cntg + clny | exp_year + imp_year,
    trade_panel
  )

  m2 <- fepoisson(
    trade ~ log_dist + lang + cntg + clny | exp_year + imp_year | pair,
    trade_panel
  )

  v1 <- vcov(m1)
  v2 <- vcov(m2, type = "clustered")
  v3 <- vcov(m1, type = "sandwich")
  v4 <- vcov(m1, type = "outer.product")

  expect_gt(norm(v2), norm(v1))
  expect_gt(norm(v3), norm(v1))
  expect_gt(norm(v4), norm(v1))
})
