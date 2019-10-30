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


///
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

#include "matrixView.hpp"
#include "index.hpp"

//[[Rcpp::export]]
Rcpp::List matrixTest(std::vector<double> &yVec, const std::vector<int32_t> &fVec, const int32_t &d, const int32_t &idx){
	try {
		BayesicSpace::MatrixView first(&yVec, 0, d, d);
		std::vector<size_t> fac;
		for (auto &i : fVec) {
			if (i < 0) {
				Rcpp::stop("Factor elements cannot be negative");
			}
			fac.push_back(static_cast<size_t>(i));
		}
		BayesicSpace::Index ind(fac);
		std::vector<double> res(2*d*d, 0.0);
		BayesicSpace::MatrixView third(&res, 0, d, 2*d);

		first.colExpand(ind, third);
		return Rcpp::List::create(Rcpp::Named("res", res));
	} catch(std::string problem) {
		Rcpp::stop(problem);
	}

	return Rcpp::List::create(Rcpp::Named("error", "NaN"));
}

