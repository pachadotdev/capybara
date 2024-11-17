#' srr_stats
#'
#' @srrstatsVerbose TRUE
#'
#' @srrstats {G1.5} Software should include all code necessary to reproduce results which form the basis of performance claims made in associated publications.
#' @srrstats {G1.6} Software should include code necessary to compare performance claims with alternative implementations in other R packages.
#' @srrstats {G2.0a} Provide explicit secondary documentation of any expectations on lengths of inputs
#' @srrstats {G2.1} Implement assertions on types of inputs (see the initial point on nomenclature above).
#' @srrstats {G2.1a} Provide explicit secondary documentation of expectations on data types of all vector inputs.
#' @srrstats {G2.2} Appropriately prohibit or restrict submission of multivariate input to parameters expected to be univariate.
#' @srrstats {G2.4} Provide appropriate mechanisms to convert between different data types, potentially including:
#' @srrstats {G2.4a} explicit conversion to `integer` via `as.integer()`
#' @srrstats {G2.4b} explicit conversion to continuous via `as.numeric()`
#' @srrstats {G2.4c} explicit conversion to character via `as.character()` (and not `paste` or `paste0`)
#' @srrstats {G2.4d} explicit conversion to factor via `as.factor()`
#' @srrstats {G2.4e} explicit conversion from factor via `as...()` functions
#' @srrstats {G2.5} Where inputs are expected to be of `factor` type, secondary documentation should explicitly state whether these should be `ordered` or not, and those inputs should provide appropriate error or other routines to ensure inputs follow these expectations.
#' @srrstats {G2.6} Software which accepts one-dimensional input should ensure values are appropriately pre-processed regardless of class structures.
#' @srrstats {G2.7} Software should accept as input as many of the above standard tabular forms as possible, including extension to domain-specific forms.
#' @srrstats {G2.8} Software should provide appropriate conversion or dispatch routines as part of initial pre-processing to ensure that all other sub-functions of a package receive inputs of a single defined class or type.
#' @srrstats {G2.9} Software should issue diagnostic messages for type conversion in which information is lost (such as conversion of variables from factor to character; standardisation of variable names; or removal of meta-data such as those associated with [`sf`-format](https://r-spatial.github.io/sf/) data) or added (such as insertion of variable or column names where none were provided).
#' @srrstats {G2.10} Software should ensure that extraction or filtering of single columns from tabular inputs should not presume any particular default behaviour, and should ensure all column-extraction operations behave consistently regardless of the class of tabular data used as input.
#' @srrstats {G2.11} Software should ensure that `data.frame`-like tabular objects which have columns which do not themselves have standard class attributes (typically, `vector`) are appropriately processed, and do not error without reason. This behaviour should be tested. Again, columns created by the [`units` package](https://github.com/r-quantities/units/) provide a good test case.
#' @srrstats {G2.12} Software should ensure that `data.frame`-like tabular objects which have list columns should ensure that those columns are appropriately pre-processed either through being removed, converted to equivalent vector columns where appropriate, or some other appropriate treatment such as an informative error. This behaviour should be tested.
#' @srrstats {G2.13} Statistical Software should implement appropriate checks for missing data as part of initial pre-processing prior to passing data to analytic algorithms.
#' @srrstats {G2.14} Where possible, all functions should provide options for users to specify how to handle missing (`NA`) data, with options minimally including:
#' @srrstats {G2.14a} error on missing data
#' @srrstats {G2.14b} ignore missing data with default warnings or messages issued
#' @srrstats {G2.15} Functions should never assume non-missingness, and should never pass data with potential missing values to any base routines with default `na.rm = FALSE`-type parameters (such as [`mean()`](https://stat.ethz.ch/R-manual/R-devel/library/base/html/mean.html), [`sd()`](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/sd.html) or [`cor()`](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/cor.html)).
#' @srrstats {G3.0} Statistical software should never compare floating point numbers for equality. All numeric equality comparisons should either ensure that they are made between integers, or use appropriate tolerances for approximate equality.
#' @srrstats {G3.1} Statistical software which relies on covariance calculations should enable users to choose between different algorithms for calculating covariances, and should not rely solely on covariances from the `stats::cov` function.
#' @srrstats {G3.1a} The ability to use arbitrarily specified covariance methods should be documented (typically in examples or vignettes).
#' @srrstats {G5.2a} Every message produced within R code by `stop()`, `warning()`, `message()`, or equivalent should be unique
#' @srrstats {G5.4a} For new methods, it can be difficult to separate out correctness of the method from the correctness of the implementation, as there may not be reference for comparison. In this case, testing may be implemented against simple, trivial cases or against multiple implementations such as an initial R implementation compared with results from a C/C++ implementation.
#' @srrstats {G5.4c} Where applicable, stored values may be drawn from published paper outputs when applicable and where code from original implementations is not available
#' @srrstats {RE4.8} Response variables, and associated "metadata" where applicable.
#' @srrstats {RE5.0} Scaling relationships between sizes of input data (numbers of observations, with potential extension to numbers of variables/columns) and speed of algorithm.
#' @noRd
NULL

#' NA_standards
#'
#' Any non-applicable standards can have their tags changed from `@srrstatsTODO`
#' to `@srrstatsNA`, and placed together in this block, along with explanations
#' for why each of these standards have been deemed not applicable.
#' (These comments may also be deleted at any time.)
#' @srrstatsNA {G2.14c} replace missing data with appropriately imputed values
#' @srrstatsNA {G2.16} All functions should also provide options to handle undefined values (e.g., `NaN`, `Inf` and `-Inf`), including potentially ignoring or removing such values.
#' @srrstatsNA {G4.0} Statistical Software which enables outputs to be written to local files should parse parameters specifying file names to ensure appropriate file suffices are automatically generated where not provided.
#' @srrstatsNA {G5.3} For functions which are expected to return objects containing no missing (`NA`) or undefined (`NaN`, `Inf`) values, the absence of any such values in return objects should be explicitly tested.
#' @srrstatsNA {G5.4} Correctness tests to test that statistical algorithms produce expected results to some fixed test data sets (potentially through comparisons using binding frameworks such as [RStata](https://github.com/lbraglia/RStata)).
#' @srrstatsNA {G5.5} Correctness tests should be run with a fixed random seed
#' @srrstatsNA {G5.9b} Running under different random seeds or initial conditions does not meaningfully change results
#' @srrstatsNA {G5.10} Extended tests should included and run under a common framework with other tests but be switched on by flags such as as a `<MYPKG>_EXTENDED_TESTS="true"` environment variable. - The extended tests can be then run automatically by GitHub Actions for example by adding the following to the `env` section of the workflow:
#' @srrstatsNA {G5.11} Where extended tests require large data sets or other assets, these should be provided for downloading and fetched as part of the testing workflow.
#' @srrstatsNA {G5.11a} When any downloads of additional data necessary for extended tests fail, the tests themselves should not fail, rather be skipped and implicitly succeed with an appropriate diagnostic message.
#' @srrstatsNA {RE2.2} Regression Software should provide different options for processing missing values in predictor and response data. For example, it should be possible to fit a model with no missing predictor data in order to generate values for all associated response points, even where submitted response values may be missing.
#' @srrstatsNA {RE4.1} Regression Software may enable an ability to generate a model object without actually fitting values. This may be useful for controlling batch processing of computationally intensive fitting algorithms.
#' @srrstatsNA {RE4.16} Regression Software which models distinct responses for different categorical groups should include the ability to submit new groups to `predict()` methods.
#' @srrstatsNA {RE4.14} Where possible, values should also be provided for extrapolation or forecast errors.
#' @srrstatsNA {RE4.15} Sufficient documentation and/or testing should be provided to demonstrate that forecast errors, confidence intervals, or equivalent values increase with forecast horizons.
#' @srrstatsNA {RE4.17} Model objects returned by Regression Software should implement or appropriately extend a default `print` method which provides an on-screen summary of model (input) parameters and (output) coefficients.
#' @srrstatsNA {RE6.0} Model objects returned by Regression Software (see RE4) should have default `plot` methods, either through explicit implementation, extension of methods for existing model objects, or through ensuring default methods work appropriately.
#' @srrstatsNA {RE6.1} Where the default `plot` method is NOT a generic `plot` method dispatched on the class of return objects (that is, through an S3-type `plot.<myclass>` function or equivalent), that method dispatch (or equivalent) should nevertheless exist in order to explicitly direct users to the appropriate function.
#' @srrstatsNA {RE6.2} The default `plot` method should produce a plot of the `fitted` values of the model, with optional visualisation of confidence intervals or equivalent.
#' @srrstatsNA {RE6.3} Where a model object is used to generate a forecast (for example, through a `predict()` method), the default `plot` method should provide clear visual distinction between modelled (interpolated) and forecast (extrapolated) values.
#' @noRd
NULL
