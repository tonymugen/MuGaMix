/*
 * Copyright (c) 2019 Anthony J. Greenberg
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
 * IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */


/// R interface functions
/** \file
 * \author Anthony J. Greenberg
 * \copyright Copyright (c) 2019 Anthony J. Greenberg
 * \version 1.0
 *
 *
 */

#include <bits/stdint-intn.h>
#include <cstddef>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <limits>

#include <Rcpp.h>
#include <Rcpp/Named.h>
#include <Rcpp/exceptions/cpp11/exceptions.h>

#include "gmmvb.hpp"

//' Variational Bayes model fit
//'
//' Fits a Gaussian mixture model using variational Bayes. Assumes no missing data.
//'
//' @param yVec    vectorized data matrix
//' @param d       number of traits
//' @param nGroups number of groups
//' @param alphaPr prior group size
//' @param sigSqPr prior variance
//' @param covRatio population to error covariance ratio
//' @param nReps   number of model fit attempts before picking the best fit
//' @return list containing group means (\code{groupMeans}), covariances (\code{covariances}), effective group sizes (\code{effNm}), group assignment probabilities (\code{p}), and the deviance information criterion (DIC, \code{DIC}).
//'
//' @keywords internal
//'
//[[Rcpp::export(name="vbFit")]]
Rcpp::List vbFit(const std::vector<double> &yVec, const int32_t &d, const int32_t &nGroups, const double &alphaPr, const double &sigSqPr, const double &covRatio, const int32_t nReps){
	if (nGroups <= 1) {
		Rcpp::stop("Number of groups must be greater than 1");
	}
	if (d <= 0) {
		Rcpp::stop("Number of traits must be non-negative");
	}
	if (nReps <= 0) {
		Rcpp::stop("Number of replicate runs must be positive");
	}
	if (alphaPr <= 0.0) {
		Rcpp::stop("Prior group size must be positive");
	}
	if (sigSqPr <= 0.0) {
		Rcpp::stop("Prior variance must be positive");
	}
	if (covRatio <= 0.0) {
		Rcpp::stop("Variance ratio must be positive");
	}
	std::vector<double> vGrpMn;
	std::vector<double> vSm;
	std::vector<double> Nm;
	std::vector<double> r;
	std::vector<double> lPost;
	double dic = 0.0;
	try {
		BayesicSpace::GmmVB vbModel(&yVec, covRatio, sigSqPr, alphaPr, static_cast<size_t>(nGroups), static_cast<size_t>(d), &vGrpMn, &vSm, &r, &Nm);
		vbModel.fitModel(lPost, dic);
		for (int iRep = 1; iRep < nReps; iRep++) {
			std::vector<double> vGrpMnLoc;
			std::vector<double> vSmLoc;
			std::vector<double> NmLoc;
			std::vector<double> rLoc;
			std::vector<double> lPostLoc;
			double dicLoc = 0.0;

			BayesicSpace::GmmVB vbModel(&yVec, covRatio, sigSqPr, alphaPr, static_cast<size_t>(nGroups), static_cast<size_t>(d), &vGrpMnLoc, &vSmLoc, &rLoc, &NmLoc);
			vbModel.fitModel(lPostLoc, dicLoc);
			if (dicLoc < dic) { // if we found a better DIC
				vGrpMn = vGrpMnLoc;
				vSm    = vSmLoc;
				Nm     = NmLoc;
				r      = rLoc;
				dic    = dicLoc;
			}
		}
	} catch (std::string problem) {
		Rcpp::stop(problem);
	}
	return Rcpp::List::create(Rcpp::Named("groupMeans", vGrpMn), Rcpp::Named("covariances", vSm), Rcpp::Named("effNm", Nm), Rcpp::Named("p", r), Rcpp::Named("DIC", dic));
}

//' Variational Bayes model fit with missing data
//'
//' Fits a Gaussian mixture model using variational Bayes. Allows missing data. Missing values should be marked with \code{NaN}.
//'
//' @param yVec     vectorized data matrix
//' @param d        number of traits
//' @param nGroups  number of groups
//' @param alphaPr  prior group size
//' @param sigSqPr  prior variance
//' @param covRatio population to error covariance ratio
//' @param nReps    number of model fit attempts before picking the best fit
//' @return list containing group means (\code{groupMeans}), covariances (\code{covariances}), effective group sizes (\code{effNm}), group assignment probabilities (\code{p}), and the deviance information criterion (DIC, \code{DIC}).
//'
//' @keywords internal
//'
//[[Rcpp::export(name="vbFitMiss")]]
Rcpp::List vbFitMiss(std::vector<double> &yVec, const int32_t &d, const int32_t &nGroups, const double &alphaPr, const double &sigSqPr, const double &covRatio, const int32_t nReps){
	if (nGroups <= 1) {
		Rcpp::stop("Number of groups must be greater than 1");
	}
	if (d <= 0) {
		Rcpp::stop("Number of traits must be non-negative");
	}
	if (nReps <= 0) {
		Rcpp::stop("Number of replicate runs must be positive");
	}
	if (alphaPr <= 0.0) {
		Rcpp::stop("Prior group size must be positive");
	}
	if (sigSqPr <= 0.0) {
		Rcpp::stop("Prior variance must be positive");
	}
	if (covRatio <= 0.0) {
		Rcpp::stop("Variance ratio must be positive");
	}
	std::vector<double> vGrpMn;
	std::vector<double> vSm;
	std::vector<double> Nm;
	std::vector<double> r;
	std::vector<double> lPost;
	double dic = 0.0;
	try {
		BayesicSpace::GmmVBmiss vbModel(&yVec, covRatio, sigSqPr, alphaPr, static_cast<size_t>(nGroups), static_cast<size_t>(d), &vGrpMn, &vSm, &r, &Nm);
		vbModel.fitModel(lPost, dic);
		for (int iRep = 1; iRep < nReps; iRep++) {
			std::vector<double> vGrpMnLoc;
			std::vector<double> vSmLoc;
			std::vector<double> NmLoc;
			std::vector<double> rLoc;
			std::vector<double> lPostLoc;
			double dicLoc = 0.0;

			BayesicSpace::GmmVBmiss vbModel(&yVec, covRatio, sigSqPr, alphaPr, static_cast<size_t>(nGroups), static_cast<size_t>(d), &vGrpMnLoc, &vSmLoc, &rLoc, &NmLoc);
			vbModel.fitModel(lPostLoc, dicLoc);
			if (dicLoc < dic) { // if we found a better DIC
				vGrpMn = vGrpMnLoc;
				vSm    = vSmLoc;
				Nm     = NmLoc;
				r      = rLoc;
				dic    = dicLoc;
			}
		}
	} catch (std::string problem) {
		Rcpp::stop(problem);
	}
	return Rcpp::List::create(Rcpp::Named("groupMeans", vGrpMn), Rcpp::Named("covariances", vSm), Rcpp::Named("effNm", Nm), Rcpp::Named("p", r), Rcpp::Named("DIC", dic));
}



