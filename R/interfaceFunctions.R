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

#' Fit a Gaussian mixture model using variational Bayes
#'
#' Quick approximate fit to a Gaussian mixture model using variational Bayes. Missing data are allowed, but no replicate measurements on individuals.
#' The model shrinks small groups to zero if the \code{priorGroupSize} parameter is small.
#' The variational Bayes algorithm sometimes gets stuck in local maxima. It is therefore a good idea to run the model fit several times and pick the "best" result.
#' The \code{nReps} parameter controls the number of replicate runs. The result with the highest value of the deviance information criterion (DIC) is reported as the best.
#'
#' @param data data frame with the data
#' @param traitColumns list of columns in \code{data} that contain trait values
#' @param nGroups number of groups to fit, must be two or greater
#' @param priorGroupSize prior group size; default value 10^-3^
#' @param nReps number of runs to do before picking the best; default value 5
#' @return S3 object of class \code{vbmugamix} that contains a matrix of group means (\code{groupMeans}), a list of within-group covariance matrices (\code{covariances}), a vector of effective group sizes (\code{effNm}), a matrix of group assignment probabilities (\code{p}; individuals in rows groups in columns), and the DIC value (\code{DIC})
#'
#' @export
quickFitModel <- function(data, traitColumns, nGroups, priorGroupSize = 1e-3, nReps = 5) {
	d <- length(traitColumns)
	if (d <= 1) {
		stop("Must have at least two traits")
	}
	yVec <- as.double(unlist(data[, traitColumns]))
	if (nGroups < 2) {
		stop("Must specify more than one group")
	}
	tau0    <- 1.0
	lambda0 <- 1.0
	res     <- NULL
	if (any(is.na(yVec))) {
		yVec[is.na(yVec)] <- NaN
		res <- MuGaMix::vbFitMiss(yVec, d, nGroups, priorGroupSize, tau0, lambda0, nReps)
	} else {
		res <- MuGaMix::vbFit(yVec, d, nGroups, priorGroupSize, tau0, lambda0, nReps)
	}
	res$p           <- matrix(res$p, ncol = nGroups)
	res$groupMeans  <- matrix(res$groupMeans, nrow = nGroups)
	covFac          <- rep(1:nGroups, each = d^2)
	res$covariances <- tapply(res$covariances, covFac, matrix, nrow = d)
	class(res)      <- "vbmugamix"
	return(res)
}

#' Fit a Gaussian mixture model using MCMC
#'
#' Fits a mixture model to the provided data, taking into account replication structure. Takes data on multiple traits and generates samples from posterior distributions of parameters, as well as probabilities that a line belongs to a given population. The fit is performed using a No-U-Turn Sampler (NUTS). The recommended burn-in, sampling, number of chains, and thinning are set as defaults.
#'
#' Missing phenotype data are allowed. It is recommended that rows with all trait data missing are eliminated before running the function. If such rows are present, they will be eliminated by the function and a warning is issued. If all data for a given line are missing, this line is also dropped from consideration. The function does its best to retain the user provided factor level ordering, however such behavior may not be desired by some users, hence the warning. The (potentially modified) line factor is returned by the function to aid in trouble shooting.
#'
#' @param data data frame with the data
#' @param trait.columns list of columns in \code{data} that contain trait values
#' @param n.pop number of populations to fit, must be two or greater
#' @param factor.column name of the column that contains the factor connecting replicates to lines (if omitted, no replication is assumed)
#' @param n.burnin number of iterations of burnin (adaptation)
#' @param n.sampling number of sampling steps
#' @param n.thin thinning number (if, e.g., set to five then every fifth chain sample is saved)
#' @param n.chains number of chains
#' @return S3 object of class \code{mugamix} that contains matrix of parameter chains (named \code{thetaChain}, each chain a column), a matrix of population assignments (named \code{piChain}), a matrix of inverse-covariances (named \code{iSigChain}), a matrix of imputed missing data (if any; named \code{imputed}), the list of lines used in model fitting (ordered the same as in the output and possibly modified from the user's input if missing rows are eliminated; named \code{lineIDs}), the number of retained samples per chain (named \code{n.samples}), and the number of population specified in the model (named \code{n.pops})
#'
#' @export
fitModel <- function(data, trait.columns, n.pop, factor.column = NULL,
				n.burnin = 5000, n.sampling = 10000, n.thin = 5, n.chains = 5) {
	d    <- length(trait.columns)
	if (d <= 1) {
		stop("Must have at least two traits")
	}
	yVec <- as.double(unlist(data[, trait.columns]))
	if (n.pop < 2) {
		stop("Must specify more than one population")
	}
	if (is.null(factor.column)) { # no replication
		if (any(is.na(yVec))) {
			stop("Missing data with no replication not supported yet")
		} else {
			res            <- runSamplerNR(yVec, d, n.pop, n.burnin, n.sampling, n.thin, n.chains)
			res$thetaChain <- matrix(res$thetaChain, ncol = n.chains)
			res$piChain    <- matrix(res$piChain, ncol = n.chains)
			res$iSigChain  <- matrix(res$iSigChain, ncol = n.chains)
			res$imputed    <- NULL
			res$lineIDs    <- rownames(data)
			res$n.samples  <- n.sampling / n.thin
			res$n.pops     <- n.pop
			class(res)     <- "mugamix"
			return(res)
		}
	} else {
		if (is.factor(data[, factor.column])) {
			lnFac <- data[, factor.column]
			lnInd <- as.integer(lnFac)
		} else {
			lnFac <- factor(data[, factor.column], levels = unique(data[, factor.column]))
			lnInd <- as.integer(lnFac)
		}
		if (any(is.na(yVec))) {
			missRowCount <- apply(data[,trait.columns], 1, function(vec) {sum(is.na(vec))})
			d            <- length(trait.columns)
			if (any(missRowCount == d)) {
				warning("WARNING: sime rows have only missing data; deleting them. This may result in loss of some lines")
				data <- data[-which(missRowCount == d), ]
				# re-define the line factor, since there is no guarantee every line has data
				if (is.factor(data[, factor.column])) {
					oldLev <- levels(data[, factor.column]) # want to preserve the user's level order
					lnFac  <- as.character(data[, factor.column])
					lnFac  <- factor(lnFac, levels = oldLev[oldLev %in% unique(lnFac)])
					lnInd  <- as.integer(lnFac)
				} else {
					lnFac <- factor(data[, factor.column], levels = unique(data[, factor.column]))
					lnInd <- as.integer(lnFac)
				}
			}
			missInd <- rep(0, times = length(yVec))
			missInd[which(is.na(yVec))] <- 1
			yVec[which(missInd == 1)]   <- 0.0 # lazy "imputation"; the right thing will be done by the model
			res            <- runSamplerMiss(yVec, lnInd, as.integer(missInd), n.pop, n.burnin, n.sampling, n.thin, n.chains)
			res$thetaChain <- matrix(res$thetaChain, ncol = n.chains)
			res$piChain    <- matrix(res$piChain, ncol = n.chains)
			res$iSigChain  <- matrix(res$iSigChain, ncol = n.chains)
			res$imputed    <- matrix(res$imputed, ncol = n.chains)
			res$lineIDs    <- levels(lnFac)
			res$n.samples  <- n.sampling / n.thin
			res$n.pops     <- n.pop
			class(res)     <- "mugamix"
			return(res)
		} else {
			res            <- runSampler(yVec, lnInd, n.pop, n.burnin, n.sampling, n.thin, n.chains)
			res$thetaChain <- matrix(res$thetaChain, ncol = n.chains)
			res$piChain    <- matrix(res$piChain, ncol = n.chains)
			res$iSigChain  <- matrix(res$iSigChain, ncol = n.chains)
			res$imputed    <- NULL
			res$lineIDs    <- levels(lnFac)
			res$n.samples  <- n.sampling / n.thin
			res$n.pops     <- n.pop
			class(res)     <- "mugamix"
			return(res)
		}
	}
}

#' Plot MCMC population assignments
#'
#' Plot method for MuGaMix objects (generated by \code{fitModel()}). Plots population assignment probabilities for each line. Uses \code{data.table} and \code{ggplot2} if available.
#' If the data set contains many lines, their names may not show up well. In such cases, the user can further modify plot parameters by using the plot objects returned by the function.
#'
#' @param obj a \code{mugamix} object generated by the \code{fitModel} function.
#'
#' @return a plot object (either \code{ggplot} or \code{barplot})
#'
#' @export
plot.mugamix <- function(obj) {
	# Calculate medians across samples and chains
	nLn      <- length(obj$lineIDs)
	lnPopFac <- paste(rep(obj$lineIDs, each = obj$n.pops), rep(1:(obj$n.pops), times = nLn), sep = ".")
	lnPopFac <- factor(rep(lnPopFac, times = obj$n.samples * ncol(obj$piChain)), levels = lnPopFac)
	popP     <- matrix(tapply(array(obj$piChain), lnPopFac, median), ncol = obj$n.pops, byrow = TRUE)
	colnames(popP) <- as.character(1:obj$n.pops)
	popAss   <- NULL
	if (requireNamespace("data.table", quietly = TRUE)) { # if we have data.table installed
		popP   <- data.table::as.data.table(popP)
		data.table::set(popP, NULL, "line", obj$lineIDs)
		popP   <- data.table::setorderv(popP, as.character(1:obj$n.pops), rep(-1, obj$n.pops))
		popAss <- data.table::melt(popP, measure = as.character(1:obj$n.pops), variable.name = "population", value.name = "p")
		data.table::set(popAss, NULL, "line", factor(popAss[, "line"], levels = unique(popAss[, "line"])))
	} else {
		popP      <- as.data.frame(popP)
		popP$line <- obj$lineIDs
		popP      <- popP[do.call(order, -popP[, 1:obj$n.pops]), ]
		popAss    <- data.frame(p = unlist(popP[, 1:obj$n.pops]),
						line = factor(rep(as.character(popP[, "line"]), times = obj$n.pops),
										levels = unique(as.character(popP[, "line"]))),
						population = rep(as.character(1:obj$n.pops), each = nLn))
	}
	if (requireNamespace("ggplot2", quietly = TRUE)) {
		return(ggplot2::ggplot(data = popAss, ggplot2::aes(x = line, y = p, fill = population)) + ggplot2::geom_col())
	} else {
		popMat <- t(as.matrix(popP[, 1:obj$n.pops]))
		colnames(popMat) <- unlist(popP[, "line"])
		return(barplot(popMat, border = NA, col = rainbow(obj$n.pops), ylab = "p", las = 2))
	}
}

#' Plot variational Bayes population assignments
#'
#' Plot method for MuGaMix objects (generated by \code{quickFitModel()}). Plots population assignment probabilities for each line. Uses \code{data.table} and \code{ggplot2} if available.
#' If the data set contains many lines, their names may not show up well. In such cases, the user can further modify plot parameters by using the plot objects returned by the function.
#'
#' @param obj a \code{vbmugamix} object generated by the \code{quickFitModel} function.
#'
#' @return a plot object (either \code{ggplot} or \code{barplot})
#'
#' @export
plot.vbmugamix <- function(obj) {
	nGrp    <- ncol(obj$p)
	nLn     <- nrow(obj$p)
	indivID <- NULL
	if (is.null(row.names(obj$p))) {
		indivID <- as.character(1:nLn)
	} else {
		indivID <- row.names(obj$p)
	}
	pOrd <- NULL
	if (requireNamespace("data.table", quietly = TRUE)) { # if we have data.table installed
		pOrd <- data.table::as.data.table(obj$p)
		data.table::setnames(pOrd, paste0("G", 1:nGrp))
		data.table::set(pOrd, 1:nLn, "individual", indivID)
		bestGrp <- apply(obj$p, 1, which.max)
		data.table::set(pOrd, 1:nLn, "bestGrp", paste0("G", bestGrp))
		pOrd    <- data.table::melt(pOrd, measure = paste0("G", 1:nGrp), variable.name = "group", value.name = "p")
		pOrd    <- data.table::setorderv(pOrd, c("bestGrp", "p"), c(1, -1))
		indivS  <- factor(unlist(pOrd[, "individual"]), levels = unique(unlist(pOrd[, "individual"])))
		data.table::set(pOrd, 1:nrow(pOrd), "individualS", indivS)
	} else {
		pOrd <- data.frame(p           = array(obj$p),
							individual = indivID,
							bestGrp    = -apply(obj$p, 1, which.max),
							group      = rep(paste0("G", 1:nGrp), each = nLn))
		pOrd <- pOrd[do.call(order, -pOrd[, c("bestGrp", "p")]), ]
		pOrd$individualS <- factor(pOrd$individual, levels = unique(pOrd$individual))
	}
	if (requireNamespace("ggplot2", quietly = TRUE)) {
		return(ggplot2::ggplot(data = pOrd, ggplot2::aes(x = individualS, y = p, fill = group)) + ggplot2::geom_col())
	} else {
		grpMat <- t(obj$p)
		colnames(grpMat) <- indivID
		grpMat <- grpMat[, levels(indivS)]
		return(barplot(grpMat, border = NA, col = rainbow(nGrp), ylab = "p", las = 2))
	}
}
