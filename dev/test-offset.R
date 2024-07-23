test_that("offset works", {
  m1 <- feglm(mpg ~ wt | cyl, data = mtcars, family = poisson())
  y <- predict(m1, type = "response")
  o1 <- feglm_offset_(m1, y)

  # m2 <- alpaca::feglm(mpg ~ wt | cyl, data = mtcars, family = poisson())
  # o2 <- drop(alpaca:::feglmOffset(m2, y)
  # datapasta::vector_paste(round(o2, 4))
  o2 <- c(
    3.018703, 3.011154, 3.056387, 3.001613, 2.979713, 2.995091, 2.976723,
    3.026537, 3.027809, 2.995612, 2.995612, 2.999650, 3.006936, 3.005836,
    2.977558, 2.974679, 2.975975, 3.094682, 3.062526, 3.053450, 3.029361,
    2.956144, 2.958109, 2.949010, 2.948902, 3.049442, 3.041447, 3.066858,
    2.964431, 2.992499, 2.955002, 3.018302
  )

  expect_equal(round(o1, 4), round(o2, 4))
})
