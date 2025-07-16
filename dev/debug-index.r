# Debug index conversion issues
devtools::load_all()

# Simple single FE case first
cat("=== Testing single FE case ===\n")
data(mtcars)

# Check the data dimensions
cat("mtcars dimensions:", dim(mtcars), "\n")
cat("Unique cyl values:", unique(mtcars$cyl), "\n")
cat("Number of obs per cyl:\n")
print(table(mtcars$cyl))

# Try the simplest case - no FE first
cat("\n=== No FE case ===\n")
try({
  result_no_fe <- felm(mpg ~ wt, mtcars)
  cat("No FE - SUCCESS\n")
  cat("Coefficients:", result_no_fe$coefficients, "\n")
}, silent = FALSE)

# Try single FE
cat("\n=== Single FE case ===\n")
try({
  result_single_fe <- felm(mpg ~ wt | cyl, mtcars)
  cat("Single FE - SUCCESS\n")
  cat("Coefficients:", result_single_fe$coefficients, "\n")
  cat("Fixed effects length:", length(result_single_fe$fixed.effects), "\n")
}, silent = FALSE)

# Check what the group structure looks like before conversion
cat("\n=== Debugging group structure ===\n")
library(capybara)

# Try to see what's happening in the group creation
tryCatch({
  # Create the fixed effects structure manually
  fe_data <- model.frame(mpg ~ wt | cyl, mtcars)
  cat("Model frame created successfully\n")
  cat("FE data dimensions:", dim(fe_data), "\n")
}, error = function(e) {
  cat("Error in model.frame:", e$message, "\n")
})
