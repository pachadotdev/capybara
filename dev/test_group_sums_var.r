# the data comes from running this after saving the intermediate outputs in alpaca
# data <- simGLM(1000L, 20L, 1805L, model = "logit")
# mod <- feglm(y ~ x1 + x2 + x3 | i + t, data)
# mod.bc <- biasCorr(mod)
# mod.ape <- getAPEs(mod.bc)

devtools::load_all()

# from alpaca
inp <- readRDS("dev/groupSumsVar_inputs.rds")

names(inp)

b <- capybara:::group_sums_var_(inp$Delta, inp$k.list.2)

class(b)
class(inp$b)

all.equal(b, inp$b)
