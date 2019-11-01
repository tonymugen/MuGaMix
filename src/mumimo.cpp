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

/// Multitrait mixture models
/** \file
 * \author Anthony J. Greenberg
 * \copyright Copyright (c) 2019 Anthony J. Greenberg
 * \version 1.0
 *
 * Class implementation to generate Markov chains for inference from multitrait Gaussian mixture models. Dual-averaging NUTS and Metropolis samplers for parameters groups are included within a Gibbs sampler.
 *
 */

#include <vector>
#include <string>
#include <cmath>

#include "index.hpp"
#include "random.hpp"
#include "model.hpp"
#include "mumimo.hpp"
#include "sampler.hpp"
#include "danuts.hpp"
#include "matrixView.hpp"

using std::vector;
using std::string;
using namespace BayesicSpace;

// MumiLoc methods
MumiLoc::MumiLoc(vector<double> *yVec, vector<double> *iSigVec, const size_t &d, const vector<Index> *hierInd, vector<double> *xVec, const double &tau) : Model(), hierInd_{hierInd}, tau0_{tau}, tauP_{tau} {
	const size_t n = yVec->size()/d;
	if (n != (*hierInd_)[0].size()) {
		throw string("ERROR: Number of rows implied by yVec not the same as in the first Index in MumiLoc contructor");
	} else if (iSigVec->size() < 2*d*d) {
		throw string("ERROR: Number of elements in iSigVec not compatible with two covariance matrices in MumiLoc constructor");
	}
	Y_     = MatrixView(yVec, 0, n, d);
	X_     = MatrixView(xVec, 0, n, xVec->size()/n);
	ISigE_ = MatrixView(iSigVec, 0, d, d);
	ISigA_ = MatrixView(iSigVec, d*d, d, d);
}

double MumiLoc::logPost(const vector<double> &theta) const{
	double lnP = 0.0;
	const size_t Nln  = (*hierInd_)[0].groupNumber();
	const size_t Npop = (*hierInd_)[1].groupNumber();
	const size_t Ydim = Y_.getNrows()*Y_.getNcols();
	vector<double> thetaCopy(theta);
	MatrixView B(&thetaCopy, 0, Y_.getNrows(), Y_.getNcols());
	MatrixView A(&thetaCopy, Y_.getNrows()*Y_.getNcols(), Nln, Y_.getNcols() );
	MatrixView M(&thetaCopy, Ydim*Nln*Y_.getNcols(), Npop, Y_.getNcols() );

	// Calculate the residual Y - XB - ZA matrix
	vector<double> vResid(Ydim, 0.0);

	return lnP;
}

void MumiLoc::gradient(const vector<double> &theta, vector<double> &grad) const{

}

