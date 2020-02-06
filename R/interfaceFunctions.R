#
# Copyright (c) 2019 Anthony J. Greenberg
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.
#

#' Fit a Gaussian mixture model
#'
#' Fits a mixture model to the provided data, taking into account replication structure. Takes data on multiple traits and generates samples from posterior distributions of parameters, as well as probabilities that a line belongs to a given population. The fit is performed using a No-U-Turn Sampler (NUTS). The recommended burn-in, sampling, number of chains, and thinning are set as defaults.
#'
#' @note Currently exactly one level of replication is supported.
#'
#' @param data data frame with the data
#' @param trait.columns list of columns in \code{data} that contain trait values
#' @param factor.column name of the column that contains the factor connecting replicates to lines
#' @param n.pop number of populations to fit, must be two ro greater
#' @param n.burnin number of iterations of butnin (adaptation)
#' @param n.sampling number of sampling steps
#' @param n.thin thinning number (if, e.g., set to five then every fifth chain sample is saved)
#' @param n.chains number of chains
#' @return S3 object of class \code{mugamix} that contains matrix of parameter chains (named \code{parChains}, each chain a column), a matrix of population assignments (named \code{popChains}), and a matrix of population numbers (named \code{nPopsChain})
#'
#' @export
fitModel <- function(data, trait.colums, factor.column, n.pop, n.burnin = 5000, n.sampling = 10000, n.thin = 5, n.chains = 5){
	yVec <- as.double(unlist(data[, trait.colums]))
	if (is.factor(data[, factor.column])) {
		lnFac <- as.integer(data[, factor.column])
	} else {
		lnFac <- as.integer(factor(data[, factor.column], levels=unique(data[, factor.column])))
	}
	if (n.pop < 2) {
		stop("Must specify more than one population")
	}
	if (any(is.na(yVec))) {
		missInd <- rep(0, times=length(yVec))
		missInd[which(is.na(yVec))] <- 1
		yVec[which(missInd == 1)]   <- 0.0 # lazy imputation; the right thing will be done by the model
		res            <- runSamplerMiss(yVec, lnFac, as.integer(missInd), n.pop, n.burnin, n.sampling, n.thin, n.chains)
		res$thetaChain <- matrix(res$thetaChain, ncol=n.chains)
		res$piChain    <- matrix(res$piChain, ncol=n.chains)
		res$nPopsChain <- matrix(res$nPopsChain, ncol=n.chains)
		res$lineFactor <- lnFac
		class(res)     <- "mugamix"
		return(res)
	} else {
		res            <- runSampler(yVec, lnFac, n.pop, n.burnin, n.sampling, n.thin, n.chains)
		res$thetaChain <- matrix(res$thetaChain, ncol=n.chains)
		res$piChain    <- matrix(res$piChain, ncol=n.chains)
		res$nPopsChain <- matrix(res$nPopsChain, ncol=n.chains)
		res$lineFactor <- lnFac
		class(res)     <- "mugamix"
		return(res)
	}
}

