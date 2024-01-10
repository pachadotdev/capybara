# the data comes from running this after saving the intermediate outputs in alpaca
# data <- simGLM(1000L, 20L, 1805L, model = "logit")
# mod <- feglm(y ~ x1 + x2 + x3 | i + t, data)

devtools::load_all()

# from alpaca
inp <- readRDS("dev/centerVariables_inputs.rds")

names(inp)

MX.centered <- capybara:::center_variables_(
  as.matrix(inp$MX), as.double(inp$w),
  as.list(inp$k.list), as.double(inp$center.tol)
)

class(MX.centered)
class(inp$MX.centered)

all.equal(MX.centered, inp$MX.centered)
