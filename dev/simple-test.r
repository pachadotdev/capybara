load_all()

coef(felm(mpg ~ wt | cyl, mtcars))
coef(fixest::feols(mpg ~ wt | cyl, mtcars))

coef(felm(mpg ~ wt + hp | cyl, mtcars))
coef(fixest::feols(mpg ~ wt + hp | cyl, mtcars))

load_all()

# Test with just the demeaning to see what's happening
mtcars_test <- mtcars
mtcars_test$cyl_fe <- as.integer(as.factor(mtcars_test$cyl)) - 1

# Compare the demeaned variables directly
cat('=== Testing demeaning directly ===\n')

# Single variable case
single_result <- capybara:::demean_variables_(
  mtcars_test$wt, 
  rep(1, nrow(mtcars_test)),
  list(mtcars_test$cyl_fe),
  0.001,  # 3 unique cylinders
  max_iter = 1000L,
  iter_interrupt = 1000L,
  iter_ssr = 10L,
  family = "gaussian"
)

cat('Single variable demeaned wt (first 5):\n')
print(head(single_result[[1]], 5))

# Multi variable case
multi_result <- capybara:::demean_variables(
  list(mtcars_test$wt, mtcars_test$hp, mtcars_test$mpg), 
  rep(1, nrow(mtcars_test)),
  list(mtcars_test$cyl_fe),
  c(3),  # 3 unique cylinders
  list(c(11, 7, 14))  # counts per cylinder
)

cat('Multi variable demeaned wt (first 5):\n')
print(head(multi_result[[1]], 5))

cat('Difference in wt demeaning:\n')
print(head(single_result[[1]] - multi_result[[1]], 5))