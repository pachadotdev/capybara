# the data comes from running this after saving the intermediate outputs in alpaca
# data <- simGLM(1000L, 20L, 1805L, model = "logit")
# mod <- feglm(y ~ x1 + x2 + x3 | i + t, data)

devtools::load_all()

# from alpaca
inp <- readRDS("dev/getAlpha_inputs.rds")

names(inp)

fe.list <- capybara:::get_alpha_(as.double(inp$pi), as.list(inp$k.list), as.double(inp$alpha.tol))

class(fe.list)
class(inp$fe.list)

class(fe.list[[1]])
class(inp$fe.list[[1]])

all.equal(fe.list, inp$fe.list)
