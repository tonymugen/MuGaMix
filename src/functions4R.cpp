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

#include "Rcpp/Named.h"
#include "bayesicUtilities/index.hpp"
#include "mumimo.hpp"
#include "gmmvb.hpp"

//[[Rcpp::export(name="testLpostNR")]]
Rcpp::List testLpostNR(const std::vector<double> &yVec, const int32_t &d, const int32_t &Ngrp, std::vector<double> &theta, const std::vector<double> &lnP, const int32_t &ind, const double &lowerLimit, const double &upperLimit, const double &incr){
	const size_t i = static_cast<size_t>(ind - 1);
	double thtVal  = theta[i];
	double add     = lowerLimit;
	std::vector<double> lPost;
	try {
		BayesicSpace::MumiNR test(&yVec, &lnP, d, Ngrp, 1e-8, 2.5, 1e-8);
		while ( add <= upperLimit ){
			theta[i] = thtVal + add;
			lPost.push_back( test.logPost(theta) );
			add += incr;
		}
	} catch(std::string problem) {
		Rcpp::stop(problem);
	}
	return Rcpp::List::create(Rcpp::Named("lPost", lPost));
}
//[[Rcpp::export(name="testGradNR")]]
Rcpp::List testGradNR(const std::vector<double> &yVec, const int32_t &d, const int32_t &Ngrp, std::vector<double> &theta, const std::vector<double> &lnP, const int32_t &ind, const double &lowerLimit, const double &upperLimit, const double &incr){
	const size_t i = static_cast<size_t>(ind - 1);
	double thtVal  = theta[i];
	double add     = lowerLimit;
	std::vector<double> gradVal;
	std::vector<double> grad;
	try {
		BayesicSpace::MumiNR test(&yVec, &lnP, d, Ngrp, 1e-8, 2.5, 1e-8);
		while ( add <= upperLimit ){
			theta[i] = thtVal + add;
			test.gradient(theta, grad);
			gradVal.push_back(grad[i]);
			add += incr;
		}
	} catch(std::string problem) {
		Rcpp::stop(problem);
	}
	return Rcpp::List::create(Rcpp::Named("gradVal", gradVal));
}

//[[Rcpp::export(name="testSampler")]]
Rcpp::List testSampler(const std::vector<double> &yVec, const int32_t &d, const int32_t &Ngrp, const int32_t &Nadapt, const int32_t &Nsamp){
	const size_t pd    = static_cast<size_t>(d);
	const size_t pNgrp = static_cast<size_t>(Ngrp);
	std::vector<double> theta0;
	std::vector<double> thetaChain;
	std::vector<double> pChain;
	try {
		BayesicSpace::WrapMMM test(yVec, pd, Ngrp, 1e-3, 1e-5, 2.5, 1e-5, theta0);
		test.runSampler(Nadapt, Nsamp, 1, thetaChain, pChain);
	} catch(std::string problem) {
		Rcpp::stop(problem);
	}
	return Rcpp::List::create(Rcpp::Named("theta0", theta0), Rcpp::Named("chain", thetaChain));
}

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

//' Run the sampler with no replication
//'
//' Runs the sampler on the data assuming no fixed effects, missing trait data, or replication.
//'
//' @param yVec   vectorized data matrix
//' @param d      number of traits
//' @param Ngrp   number of populations
//' @param Nadapt number of adaptation (burn-in) steps
//' @param Nsamp  number of sampling steps
//' @param Nthin  thinning number
//'
//' @keywords internal
//'
//[[Rcpp::export(name="runSamplerNR")]]
Rcpp::List runSamplerNR(const std::vector<double> &yVec, const int32_t &d, const int32_t &Ngrp, const int32_t &Nadapt, const int32_t &Nsamp, const int32_t &Nthin, const int32_t &Nchains){
	if (d <= 1){
		Rcpp::stop("ERROR: there must be at least two traits");
	}
	const size_t dd = static_cast<size_t>(d);
	if (yVec.size()%d) {
		Rcpp::stop("ERROR: number of traits implies a non-integer number of individuals in the data vector");
	}
	if (Ngrp <= 1) {
		Rcpp::stop("ERROR: there must be at least two populations");
	}
	if (Nadapt < 0) {
		Rcpp::stop("ERROR: Number of adaptation (burn-in) steps must be non-negative");
	}
	if (Nsamp < 0) {
		Rcpp::stop("ERROR: Number of sampling steps must be non-negative");
	}
	if (Nchains <= 0) {
		Rcpp::stop("ERROR: Number of chains must be positive");
	}
	std::vector<double> thetaChain; // location parameter chain
	std::vector<double> iSigChain; // inverse-covariance chain
	std::vector<double> piChain;    // population probability chain
	const uint32_t Na = static_cast<uint32_t>(Nadapt);
	const uint32_t Ns = static_cast<uint32_t>(Nsamp);
	const uint32_t Nt = static_cast<uint32_t>(Nthin);
	const uint32_t Np = static_cast<uint32_t>(Ngrp);

	std::vector<double> test; // TODO: remove after testing is done
	try {
		for (int32_t i = 0; i < Nchains; i++) {
			BayesicSpace::WrapMMM modelObj(yVec, dd, Np, 1.2, 1e-8, 2.5, 1e-6, test);
			//modelObj.runSampler(Na, Ns, Nt, thetaChain, iSigChain, piChain);
		}
		return Rcpp::List::create(Rcpp::Named("thetaChain", thetaChain), Rcpp::Named("piChain", piChain), Rcpp::Named("iSigChain", iSigChain));
	} catch(std::string problem) {
		Rcpp::stop(problem);
	}
	return Rcpp::List::create(Rcpp::Named("thetaChain", thetaChain), Rcpp::Named("piChain", piChain), Rcpp::Named("iSigChain", iSigChain));
}
//' Run the sampler with one replication level
//'
//' Runs the sampler on the data assuming no fixed effects or missing trait data and one replication level.
//'
//' @param yVec vectorized data matrix
//' @param lnFac factor relating data points to lines
//' @param Ngrp number of populations
//' @param Nadapt number of adaptation (burn-in) steps
//' @param Nsamp number of sampling steps
//' @param Nthin thinning number
//'
//' @keywords internal
//'
//[[Rcpp::export(name="runSampler")]]
Rcpp::List runSampler(const std::vector<double> &yVec, const std::vector<int32_t> &lnFac, const int32_t &Ngrp, const int32_t &Nadapt, const int32_t &Nsamp, const int32_t &Nthin, const int32_t &Nchains){
	if (yVec.size()%lnFac.size()) {
		Rcpp::stop("ERROR: line factor length implies a non-integer number of traits in the data vector");
	}
	if (Ngrp <= 1) {
		Rcpp::stop("ERROR: there must be at least two populations");
	}
	if (Nadapt < 0) {
		Rcpp::stop("ERROR: Number of adaptation (burn-in) steps must be non-negative");
	}
	if (Nsamp < 0) {
		Rcpp::stop("ERROR: Number of sampling steps must be non-negative");
	}
	if (Nchains <= 0) {
		Rcpp::stop("ERROR: Number of chains must be positive");
	}
	std::vector<size_t> l1;
	for (auto &lf : lnFac) {
		if (lf <= 0) {
			Rcpp::stop("ERROR: all elements of the line factor must be positive");
		}
		l1.push_back( static_cast<size_t>(lf-1) );
	}
	std::vector<double> thetaChain; // location parameter chain
	std::vector<double> iSigChain; // inverse-covariance chain
	std::vector<double> piChain;    // population probability chain
	const uint32_t Na = static_cast<uint32_t>(Nadapt);
	const uint32_t Ns = static_cast<uint32_t>(Nsamp);
	const uint32_t Nt = static_cast<uint32_t>(Nthin);
	const uint32_t Np = static_cast<uint32_t>(Ngrp);

	try {
		for (int32_t i = 0; i < Nchains; i++) {
			BayesicSpace::WrapMMM modelObj(yVec, l1, Np, 1e-6, 0.1, 1e-8, 2.5, 1e-6);
			//modelObj.runSampler(Na, Ns, Nt, thetaChain, iSigChain, piChain);
		}
		return Rcpp::List::create(Rcpp::Named("thetaChain", thetaChain), Rcpp::Named("piChain", piChain), Rcpp::Named("iSigChain", iSigChain));
	} catch(std::string problem) {
		Rcpp::stop(problem);
	}
	return Rcpp::List::create(Rcpp::Named("thetaChain", thetaChain), Rcpp::Named("piChain", piChain), Rcpp::Named("iSigChain", iSigChain));
}

//' Run the sampler with missing data and one replication level
//'
//' Runs the sampler on the data assuming no fixed effects, but allowing for missing phenotype data, and one replication level.
//' The missingness indicator should have 1 for missing data points and 0 otherwise, however any non-0 value is treated as 1.
//'
//' @param yVec vectorized data matrix
//' @param lnFac factor relating data points to lines
//' @param missIDs vectorized matrix (same dimensions as data) with 1 where a data point is missing and 0 otherwise
//' @param Ngrp number of populations
//' @param Nadapt number of adaptation (burn-in) steps
//' @param Nsamp number of sampling steps
//' @param Nthin thinning number
//'
//' @keywords internal
//'
//[[Rcpp::export(name="runSamplerMiss")]]
Rcpp::List runSamplerMiss(const std::vector<double> &yVec, const std::vector<int32_t> &lnFac, const std::vector<int32_t> &missIDs, const int32_t &Ngrp, const int32_t &Nadapt, const int32_t &Nsamp, const int32_t &Nthin, const int32_t &Nchains){
	if (yVec.size()%lnFac.size()) {
		Rcpp::stop("ERROR: line factor length implies a non-integer number of traits in the data vector");
	}
	if (Ngrp <= 1) {
		Rcpp::stop("ERROR: there must be at least two populations");
	}
	if (Nadapt < 0) {
		Rcpp::stop("ERROR: Number of adaptation (burn-in) steps must be non-negative");
	}
	if (Nsamp < 0) {
		Rcpp::stop("ERROR: Number of sampling steps must be non-negative");
	}
	if (Nchains <= 0) {
		Rcpp::stop("ERROR: Number of chains must be positive");
	}
	std::vector<size_t> l1;
	for (auto &lf : lnFac) {
		if (lf <= 0) {
			Rcpp::stop("ERROR: all elements of the line factor must be positive");
		}
		l1.push_back( static_cast<size_t>(lf-1) );
	}
	std::vector<double> thetaChain;  // location parameter chain
	std::vector<double> iSigChain;  // inverse-covariance chain
	std::vector<double> piChain;     // population probability chain
	std::vector<double> yImpChain;   // imputed missing data chain

	const uint32_t Na = static_cast<uint32_t>(Nadapt);
	const uint32_t Ns = static_cast<uint32_t>(Nsamp);
	const uint32_t Nt = static_cast<uint32_t>(Nthin);
	const uint32_t Np = static_cast<uint32_t>(Ngrp);

	try {
		for (int32_t i = 0; i < Nchains; i++) {
			BayesicSpace::WrapMMM modelObj(yVec, l1, missIDs, Np, 0.01, 1.0, 1e-8, 2.5, 1e-6);
			modelObj.runSampler(Na, Ns, Nt, thetaChain, iSigChain, piChain, yImpChain);
		}
		return Rcpp::List::create(Rcpp::Named("thetaChain", thetaChain), Rcpp::Named("piChain", piChain), Rcpp::Named("iSigChain", iSigChain), Rcpp::Named("imputed", yImpChain));
	} catch(std::string problem) {
		Rcpp::stop(problem);
	}
	return Rcpp::List::create(Rcpp::Named("thetaChain", thetaChain), Rcpp::Named("piChain", piChain), Rcpp::Named("iSigChain", iSigChain), Rcpp::Named("imputed", yImpChain));
}

