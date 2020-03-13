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

#include <vector>
#include <cmath>
#include <algorithm>

#include <Rcpp.h>

#include "mumimo.hpp"

//' Run the sampler
//'
//' Runs the sampler on the data assuming no fixed effects or missing trait data and one replication level.
//'
//' @param yVec vectorized data matrix
//' @param lnFac factor relating data points to lines
//' @param Npop number of populations
//' @param Nadapt number of adaptation (burn-in) steps
//' @param Nsamp number of sampling steps
//' @param Nthin thinning number
//'
//[[Rcpp::export(name="runSampler")]]
Rcpp::List runSampler(const std::vector<double> &yVec, const std::vector<int32_t> &lnFac, const int32_t &Npop, const int32_t &Nadapt, const int32_t &Nsamp, const int32_t &Nthin, const int32_t &Nchains){
	if (yVec.size()%lnFac.size()) {
		Rcpp::stop("ERROR: line factor length implies a non-integer number of traits in the data vector");
	}
	if (Npop <= 1) {
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
	const uint32_t Np = static_cast<uint32_t>(Npop);

	try {
		for (int32_t i = 0; i < Nchains; i++) {
			BayesicSpace::WrapMMM modelObj(yVec, l1, Np, 1.0, 1e-8, 2.5, 1e-6);
			modelObj.runSampler(Na, Ns, Nt, thetaChain, iSigChain, piChain);
		}
		return Rcpp::List::create(Rcpp::Named("thetaChain", thetaChain), Rcpp::Named("piChain", piChain), Rcpp::Named("iSigChain", iSigChain));
	} catch(std::string problem) {
		Rcpp::stop(problem);
	}
	return Rcpp::List::create(Rcpp::Named("thetaChain", thetaChain), Rcpp::Named("piChain", piChain), Rcpp::Named("iSigChain", iSigChain));
}

//' Run the sampler with missing data
//'
//' Runs the sampler on the data assuming no fixed effects, but allowing for missing phenotype data, and one replication level.
//' The missingness indicator should have 1 for missing data points and 0 otherwise, however any non-0 value is treated as 1.
//'
//' @param yVec vectorized data matrix
//' @param lnFac factor relating data points to lines
//' @param missIDs vectorized matrix (same dimensions as data) with 1 where a data point is missing and 0 otherwise
//' @param Npop number of populations
//' @param Nadapt number of adaptation (burn-in) steps
//' @param Nsamp number of sampling steps
//' @param Nthin thinning number
//'
//[[Rcpp::export(name="runSamplerMiss")]]
Rcpp::List runSamplerMiss(const std::vector<double> &yVec, const std::vector<int32_t> &lnFac, const std::vector<int32_t> &missIDs, const int32_t &Npop, const int32_t &Nadapt, const int32_t &Nsamp, const int32_t &Nthin, const int32_t &Nchains){
	if (yVec.size()%lnFac.size()) {
		Rcpp::stop("ERROR: line factor length implies a non-integer number of traits in the data vector");
	}
	if (Npop <= 1) {
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
	const uint32_t Np = static_cast<uint32_t>(Npop);

	try {
		for (int32_t i = 0; i < Nchains; i++) {
			BayesicSpace::WrapMMM modelObj(yVec, l1, missIDs, Np, 1.0, 1e-8, 2.5, 1e-6);
			modelObj.runSampler(Na, Ns, Nt, thetaChain, iSigChain, piChain, yImpChain);
		}
		return Rcpp::List::create(Rcpp::Named("thetaChain", thetaChain), Rcpp::Named("piChain", piChain), Rcpp::Named("iSigChain", iSigChain), Rcpp::Named("imputed", yImpChain));
	} catch(std::string problem) {
		Rcpp::stop(problem);
	}
	return Rcpp::List::create(Rcpp::Named("thetaChain", thetaChain), Rcpp::Named("piChain", piChain), Rcpp::Named("iSigChain", iSigChain), Rcpp::Named("imputed", yImpChain));
}

