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
#include "index.hpp"

//[[Rcpp::export]]
double lpTestL(const std::vector<double> &yVec, const std::vector<double> &iSigVec, const std::vector<int32_t> &repFac, const std::vector<int32_t> &lnFac, const std::vector<double> &paramValues, const int32_t &d){
	if (d <= 0) {
		Rcpp::stop("ERROR: number of traits must be positive");
	}
	std::vector<size_t> l1;
	std::vector<size_t> l2;
	for (auto &rf : repFac) {
		if (rf <= 0) {
			Rcpp::stop("ERROR: all elements of the replicate factor must be positive");
		}
		l1.push_back( static_cast<size_t>(rf-1) );
	}
	for (auto &lf : lnFac) {
		if (lf <= 0) {
			Rcpp::stop("ERROR: all elements of the line factor must be positive");
		}
		l2.push_back( static_cast<size_t>(lf-1) );
	}
	try {
		std::vector<BayesicSpace::Index> factors;
		factors.push_back(BayesicSpace::Index(l1));
		factors.push_back(BayesicSpace::Index(l2));
		BayesicSpace::MumiLoc test(&yVec, &iSigVec, &factors, 1e-5);
		double res = test.logPost(paramValues);
		return res;
	} catch(std::string problem) {
		Rcpp::stop(problem);
	}

	return 0.0;
}

//[[Rcpp::export]]
Rcpp::List lpTestLI(const std::vector<double> &yVec, const std::vector<double> &iSigVec, const std::vector<int32_t> &repFac, const std::vector<int32_t> &lnFac, const std::vector<double> &paramValues, const int32_t &i, const int32_t &d, const double &mar){
	if (d <= 0) {
		Rcpp::stop("ERROR: number of traits must be positive");
	}
	std::vector<size_t> l1;
	std::vector<size_t> l2;
	for (auto &rf : repFac) {
		if (rf <= 0) {
			Rcpp::stop("ERROR: all elements of the replicate factor must be positive");
		}
		l1.push_back( static_cast<size_t>(rf-1) );
	}
	for (auto &lf : lnFac) {
		if (lf <= 0) {
			Rcpp::stop("ERROR: all elements of the line factor must be positive");
		}
		l2.push_back( static_cast<size_t>(lf-1) );
	}
	std::vector<double> lpost;
	std::vector<double> chParam(paramValues);
	try {
		std::vector<BayesicSpace::Index> factors;
		factors.push_back(BayesicSpace::Index(l1));
		factors.push_back(BayesicSpace::Index(l2));
		BayesicSpace::MumiLoc test(&yVec, &iSigVec, &factors, 1e-5);
		double val = chParam[i - 1];
		for (double add = -mar; add  <= mar; add += 0.1) {
			chParam[i - 1] = val + add;
			lpost.push_back(test.logPost(chParam));
		}
		return Rcpp::List::create(Rcpp::Named("lpost", lpost));
	} catch(std::string problem) {
		Rcpp::stop(problem);
	}

	return Rcpp::List::create(Rcpp::Named("lpost", lpost));
}

//[[Rcpp::export]]
double gradTestL(const std::vector<double> &yVec, const std::vector<double> &iSigVec, const std::vector<int32_t> &repFac, const std::vector<int32_t> &lnFac, const std::vector<double> &paramValues, const int32_t &d, const int32_t &idx){
	if (d <= 0) {
		Rcpp::stop("ERROR: number of traits must be positive");
	}
	std::vector<size_t> l1;
	std::vector<size_t> l2;
	for (auto &rf : repFac) {
		if (rf <= 0) {
			Rcpp::stop("ERROR: all elements of the replicate factor must be positive");
		}
		l1.push_back( static_cast<size_t>(rf-1) );
	}
	for (auto &lf : lnFac) {
		if (lf <= 0) {
			Rcpp::stop("ERROR: all elements of the line factor must be positive");
		}
		l2.push_back( static_cast<size_t>(lf-1) );
	}
	try {
		std::vector<BayesicSpace::Index> factors;
		factors.push_back(BayesicSpace::Index(l1));
		factors.push_back(BayesicSpace::Index(l2));
		std::vector<double> grad(yVec.size(), 0.0);
		BayesicSpace::MumiLoc test(&yVec, &iSigVec, &factors, 1e-5);
		test.gradient(paramValues, grad);
		return grad[idx-1];
	} catch(std::string problem) {
		Rcpp::stop(problem);
	}

	return 0.0;
}

//[[Rcpp::export]]
Rcpp::List gradTestLI(const std::vector<double> &yVec, const std::vector<double> &iSigVec, const std::vector<int32_t> &repFac, const std::vector<int32_t> &lnFac, const std::vector<double> &paramValues, const int32_t &d, const int32_t &idx, const double &mar){
	if (d <= 0) {
		Rcpp::stop("ERROR: number of traits must be positive");
	}
	std::vector<size_t> l1;
	std::vector<size_t> l2;
	for (auto &rf : repFac) {
		if (rf <= 0) {
			Rcpp::stop("ERROR: all elements of the replicate factor must be positive");
		}
		l1.push_back( static_cast<size_t>(rf-1) );
	}
	for (auto &lf : lnFac) {
		if (lf <= 0) {
			Rcpp::stop("ERROR: all elements of the line factor must be positive");
		}
		l2.push_back( static_cast<size_t>(lf-1) );
	}
	std::vector<double> grdRes;
	std::vector<double> chParam(paramValues);
	try {
		std::vector<BayesicSpace::Index> factors;
		factors.push_back(BayesicSpace::Index(l1));
		factors.push_back(BayesicSpace::Index(l2));
		std::vector<double> grad(yVec.size(), 0.0);
		BayesicSpace::MumiLoc test(&yVec, &iSigVec, &factors, 1e-5);
		for (double add = -mar; add <= mar; add += 0.1) {
			chParam[idx-1] = paramValues[idx-1] + add;
			test.gradient(chParam, grad);
			grdRes.push_back(grad[idx-1]);
		}
		return Rcpp::List::create(Rcpp::Named("gradVal", grdRes));
	} catch(std::string problem) {
		Rcpp::stop(problem);
	}

	return Rcpp::List::create(Rcpp::Named("gradVal", grdRes));
}

//[[Rcpp::export]]
double lpTestS(const std::vector<double> &yVec, const std::vector<double> &iSigVec, const std::vector<int32_t> &repFac, const std::vector<int32_t> &lnFac, const std::vector<double> &paramValues, const int32_t &d){
	if (d <= 0) {
		Rcpp::stop("ERROR: number of traits must be positive");
	}
	std::vector<size_t> l1;
	std::vector<size_t> l2;
	for (auto &rf : repFac) {
		if (rf <= 0) {
			Rcpp::stop("ERROR: all elements of the replicate factor must be positive");
		}
		l1.push_back( static_cast<size_t>(rf-1) );
	}
	for (auto &lf : lnFac) {
		if (lf <= 0) {
			Rcpp::stop("ERROR: all elements of the line factor must be positive");
		}
		l2.push_back( static_cast<size_t>(lf-1) );
	}
	try {
		std::vector<BayesicSpace::Index> factors;
		factors.push_back(BayesicSpace::Index(l1));
		factors.push_back(BayesicSpace::Index(l2));
		BayesicSpace::MumiISig test(&yVec, &paramValues, &factors, 2.0, 1e-10);
		double res = test.logPost(iSigVec);
		//double res = 1.0;
		return res;
	} catch(std::string problem) {
		Rcpp::stop(problem);
	}

	return 0.0;
}

//[[Rcpp::export]]
double gradTestS(const std::vector<double> &yVec, const std::vector<double> &iSigVec, const std::vector<int32_t> &repFac, const std::vector<int32_t> &lnFac, const std::vector<double> &paramValues, const int32_t &d, const int32_t &idx){
	if (d <= 0) {
		Rcpp::stop("ERROR: number of traits must be positive");
	}
	std::vector<size_t> l1;
	std::vector<size_t> l2;
	for (auto &rf : repFac) {
		if (rf <= 0) {
			Rcpp::stop("ERROR: all elements of the replicate factor must be positive");
		}
		l1.push_back( static_cast<size_t>(rf-1) );
	}
	for (auto &lf : lnFac) {
		if (lf <= 0) {
			Rcpp::stop("ERROR: all elements of the line factor must be positive");
		}
		l2.push_back( static_cast<size_t>(lf-1) );
	}
	try {
		std::vector<BayesicSpace::Index> factors;
		factors.push_back(BayesicSpace::Index(l1));
		factors.push_back(BayesicSpace::Index(l2));
		std::vector<double> grad(yVec.size(), 0.0);
		BayesicSpace::MumiISig test(&yVec, &paramValues, &factors, 2.0, 1e-10);
		test.gradient(iSigVec, grad);
		return grad[idx-1];
	} catch(std::string problem) {
		Rcpp::stop(problem);
	}

	return 0.0;
}

//[[Rcpp::export]]
Rcpp::List testInitTheta(const std::vector<double> &yVec, const std::vector<int32_t> &lnFac, const std::vector<int32_t> &popFac, const int32_t &Npop, const int32_t &d){
	if (d <= 0) {
		Rcpp::stop("ERROR: number of traits must be positive");
	}
	std::vector<size_t> l1;
	std::vector<size_t> l2;
	for (auto &lf : lnFac) {
		if (lf <= 0) {
			Rcpp::stop("ERROR: all elements of the line factor must be positive");
		}
		l1.push_back( static_cast<size_t>(lf-1) );
	}
	for (auto &pf : popFac) {
		if (pf <= 0) {
			Rcpp::stop("ERROR: all elements of the population factor must be positive");
		}
		l2.push_back( static_cast<size_t>(pf-1) );
	}
	std::vector<double> theta;
	std::vector<double> iSg;
	try {
		BayesicSpace::WrapMMM test(yVec, l1, l2, 1e-5, 2.0, 1e-10);
		test.getTheta(theta);
		test.getISig(iSg);
		return Rcpp::List::create(Rcpp::Named("theta", theta), Rcpp::Named("iSig", iSg));
	} catch(std::string problem) {
		Rcpp::stop(problem);
	}
	return Rcpp::List::create(Rcpp::Named("theta", theta), Rcpp::Named("iSig", iSg));
}

//[[Rcpp::export]]
Rcpp::List testLocSampler(const std::vector<double> &yVec, const std::vector<int32_t> &lnFac, const std::vector<int32_t> &popFac, const int32_t &Npop, const int32_t &d, const int32_t &Nadapt, const int32_t &Nsamp){
	if (d <= 0) {
		Rcpp::stop("ERROR: number of traits must be positive");
	}
	if (Nadapt < 0) {
		Rcpp::stop("ERROR: Number of adaptation (burn-in) steps must be non-negative");
	}
	if (Nsamp < 0) {
		Rcpp::stop("ERROR: Number of sampling steps must be non-negative");
	}
	std::vector<size_t> l1;
	std::vector<size_t> l2;
	for (auto &lf : lnFac) {
		if (lf <= 0) {
			Rcpp::stop("ERROR: all elements of the line factor must be positive");
		}
		l1.push_back( static_cast<size_t>(lf-1) );
	}
	for (auto &pf : popFac) {
		if (pf <= 0) {
			Rcpp::stop("ERROR: all elements of the population factor must be positive");
		}
		l2.push_back( static_cast<size_t>(pf-1) );
	}
	std::vector<double> chain;
	std::vector<uint32_t> tree;
	const uint32_t Na = static_cast<uint32_t>(Nadapt);
	const uint32_t Ns = static_cast<uint32_t>(Nsamp);

	try {
		BayesicSpace::WrapMMM test(yVec, l1, l2, 1e-5, 2.0, 1e-10);
		test.runSampler(Na, Ns, chain, tree);
		return Rcpp::List::create(Rcpp::Named("chain", chain), Rcpp::Named("tree", tree));
	} catch(std::string problem) {
		Rcpp::stop(problem);
	}
	return Rcpp::List::create(Rcpp::Named("chain", chain));
}

