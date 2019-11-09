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
double lpTest(const std::vector<double> &yVec, const std::vector<double> &iSigVec, const std::vector<int32_t> &repFac, const std::vector<int32_t> &lnFac, const std::vector<double> &paramValues, const int32_t &d){
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
		std::vector<double> xVec(yVec.size()/d, 1.0); // intercept only
		BayesicSpace::MumiLoc test(&yVec, &iSigVec, d, &factors, &xVec, 1e-5);
		//double res = test.logPost(paramValues);
		double res = 1.0;
		return res;
	} catch(std::string problem) {
		Rcpp::stop(problem);
	}

	return 0.0;
}

