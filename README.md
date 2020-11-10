# Package description

This is an R package to assign individuals to groups based on phenotypic data from multiple traits using Bayesian inference. While it is geared towards biological data, any multivariate data set can be analyzed this way. The model is a Gaussian mixture, fit using variational Bayes. A previous version hosted here used  Hamiltonian Monte Carlo. This is still in development and will be added later. Goodness of fit is assessed using the deviance information criterion (DIC). The implementation is based on [McGrory and Titterington (2007)](https://www.sciencedirect.com/science/article/abs/pii/S0167947306002362?via%3Dihub)

The package is still in development. Basic model fitting with no replication and no covariates is functional. The following functionality is in development:

- Full Bayesian treatment with MCMC
- Functions to summarize results and provide convergence diagnostics
- Support for covariates
- Arbitrary replication levels

To install, make sure you have the `devtools` package on your system, and then run `install_github("tonymugen/MuGaMix")`. It should work under any Unix-like system, but has not been tested under Windows.

To fit the model, run the `quickFitModel` function (`?quickFitModel` for help). There is a plot method for the object returned by this function, which shows group assignment probabilities for each individual, similar to the plots that are used with `STRUCTURE`.

