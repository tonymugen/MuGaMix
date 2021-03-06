% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/interfaceFunctions.R
\name{quickFitModel}
\alias{quickFitModel}
\title{Fit a Gaussian mixture model using variational Bayes}
\usage{
quickFitModel(data, traitColumns, nGroups, priorGroupSize = 0.001, nReps = 5)
}
\arguments{
\item{data}{data frame with the data}

\item{traitColumns}{list of columns in \code{data} that contain trait values}

\item{nGroups}{number of groups to fit, must be two or greater}

\item{priorGroupSize}{prior group size; default value 10^-3^}

\item{nReps}{number of runs to do before picking the best; default value 5}
}
\value{
S3 object of class \code{vbmugamix} that contains a matrix of group means (\code{groupMeans}), a list of within-group covariance matrices (\code{covariances}), a vector of effective group sizes (\code{effNm}), a matrix of group assignment probabilities (\code{p}; individuals in rows groups in columns), and the DIC value (\code{DIC})
}
\description{
Quick approximate fit to a Gaussian mixture model using variational Bayes. Missing data are allowed, but no replicate measurements on individuals.
The model shrinks small groups to zero if the \code{priorGroupSize} parameter is small.
The variational Bayes algorithm sometimes gets stuck in local maxima. It is therefore a good idea to run the model fit several times and pick the "best" result.
The \code{nReps} parameter controls the number of replicate runs. The result with the highest value of the deviance information criterion (DIC) is reported as the best.
}
