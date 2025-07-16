# Test formula parsing fix
devtools::load_all()

cat("=== Testing formula parsing fix ===\n")
data(mtcars)

# Test no FE case
cat("\n=== No FE case ===\n")
try({
  result_no_fe <- felm(mpg ~ wt, mtcars)
  cat("No FE - SUCCESS\n")
  cat("Coefficients:", result_no_fe$coefficients, "\n")
}, silent = FALSE)

# Test single FE case
cat("\n=== Single FE case ===\n")
try({
  result_single_fe <- felm(mpg ~ wt | cyl, mtcars)
  cat("Single FE - SUCCESS\n")
  cat("Coefficients:", result_single_fe$coefficients, "\n")
  cat("Fixed effects length:", length(result_single_fe$fixed.effects), "\n")
  cat("Fixed effects names:", names(result_single_fe$fixed.effects), "\n")
}, silent = FALSE)

# Test binomial GLM case
cat("\n=== Binomial case ===\n")
try({
  mtcars$vs_binary <- as.numeric(mtcars$vs)
  result_binomial <- feglm(vs_binary ~ wt | cyl, mtcars, family = binomial())
  cat("Binomial FE - SUCCESS\n")
  cat("Coefficients:", result_binomial$coefficients, "\n")
}, silent = FALSE)
