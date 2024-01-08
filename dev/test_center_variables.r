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
