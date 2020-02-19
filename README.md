This is an R package to assign individuals to populations based on phenotypic data from multiple traits. It uses Bayesian inference and Hamiltonian Monte Carlo to fit the model. Goodness of fit is assessed using Bayes factors which allow to choose the most likely number of populations.

The package is still in development. Basic model fitting with one level of replication and no covariates is functional. The following functionality is in development:

 - Bayes factors to assess goodness of fit and reasonable population number
 - Functions to summarize results and provide convergence diagnostics
 - Support for covariates
 - Arbitrary replication levels (including no replication)

To install, make sure you have the `devtools` package on your system, and then run `install_github("tonymugen/MuGaMix")`. It should work under any Unix-like system, but has not been tested under Windows.

To fit the model, run the `fitModel` function (`?fitModel` for help). There is a plot method for the object returned by this function, which shows population assignment probabilities for each line, similar to the plots that are used with `STRUCTURE`.

