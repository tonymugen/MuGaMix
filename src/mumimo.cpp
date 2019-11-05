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
MumiLoc::MumiLoc(const vector<double> *yVec, const vector<double> *iSigVec, const size_t &d, const vector<Index> *hierInd, const vector<double> *xVec, const double &tau) : Model(), hierInd_{hierInd}, tau0_{tau}, tauP_{tau} {
	const size_t n = yVec->size()/d;
	if (n != (*hierInd_)[0].size()) {
		throw string("ERROR: Number of rows implied by yVec not the same as in the first Index in MumiLoc contructor");
	} else if (iSigVec->size() < 2*d*d) {
		throw string("ERROR: Number of elements in iSigVec not compatible with two covariance matrices in MumiLoc constructor");
	}
	Y_     = MatrixViewConst(yVec, 0, n, d);
	X_     = MatrixViewConst(xVec, 0, n, xVec->size()/n);
	ISigE_ = MatrixViewConst(iSigVec, 0, d, d);
	ISigA_ = MatrixViewConst(iSigVec, d*d, d, d);
}

double MumiLoc::logPost(const vector<double> &theta) const{
	double lnP = 0.0;
	const size_t Nln  = (*hierInd_)[0].groupNumber();
	const size_t Npop = (*hierInd_)[1].groupNumber();
	const size_t Ydim = Y_.getNrows()*Y_.getNcols();
	MatrixViewConst B(&theta, 0, X_.getNcols(), Y_.getNcols());
	MatrixViewConst A(&theta, X_.getNrows()*Y_.getNcols(), Nln, Y_.getNcols() );
	MatrixViewConst M(&theta, Ydim*Nln*Y_.getNcols(), Npop, Y_.getNcols() );

	// Calculate the residual Y - XB - ZA matrix
	vector<double> vResid(Ydim, 0.0);
	MatrixView mResid(&vResid, 0, Y_.getNrows(), Y_.getNcols());
	A.rowExpand((*hierInd_)[0], mResid); // ZA
	for (size_t jCol = 0; jCol  < Y_.getNcols(); ++jCol) {
		for (size_t iRow = 0; iRow < Y_.getNrows(); ++ iRow) {
			mResid.setElem(iRow, jCol, Y_.getElem(iRow, jCol) - mResid.getElem(iRow, jCol) ); // Y - ZA
		}
	}
	B.gemm(false, -1.0, X_, false, 1.0, mResid); // Y - ZA - XB
	vector<double> vResIS(Ydim, 0.0);
	MatrixView resISE(&vResIS, 0, Y_.getNrows(), Y_.getNcols());
	// Multiply the residual by inverse-SigE and take the trace of the product of that with the residual
	mResid.symm('u', 'r', 1.0, ISigE_, 0.0, resISE); // (Y-ZA-XB)Sig^-1[E]
	// need only the trace of the final product, so calculating only the diagonal elements
	double trResid = 0.0;
	for (size_t jCol = 0; jCol < Y_.getNcols(); ++jCol) {
		for (size_t iRow = 0; iRow < Y_.getNrows(); ++iRow) {
			trResid += resISE.getElem(iRow, jCol)*mResid.getElem(iRow, jCol);
		}
	}
	vResid.clear();
	vResIS.clear();
	// B cross-product trace
	double trB = 0.0;
	for (size_t jCol = 0; jCol < Y_.getNcols(); ++jCol) {
		for (size_t iRow = 0; iRow < B.getNrows(); ++iRow) {
			trB += B.getElem(iRow, jCol)*B.getElem(iRow, jCol);
		}
	}
	trB *= tau0_;
	// Now on to accession (line) mean residual (A - M[p]) and kernel trace
	vResid.resize(A.getNrows()*A.getNcols(), 0.0); // will now be the A - M residual
	vResIS.resize(A.getNrows()*A.getNcols(), 0.0); // will become (A-M)Sig^{-1}[A]
	mResid = MatrixView(&vResid, 0, A.getNrows(), A.getNcols());
	resISE = MatrixView(&vResIS, 0, A.getNrows(), A.getNcols());
	M.rowExpand((*hierInd_)[1], mResid); // Z[p]M[p]
	for (size_t jCol = 0; jCol  < A.getNcols(); ++jCol) {
		for (size_t iRow = 0; iRow < A.getNrows(); ++ iRow) {
			mResid.setElem(iRow, jCol, A.getElem(iRow, jCol) - mResid.getElem(iRow, jCol) ); // A - Z[p]M[p]
		}
	}
	mResid.symm('u', 'r', 1.0, ISigA_, 0.0, resISE); // (A-Z[p]M[p])Sig^{-1}[A]
	double trAr = 0.0;
	for (size_t jCol = 0; jCol < A.getNcols(); ++jCol) {
		for (size_t iRow = 0; iRow < A.getNrows(); ++iRow) {
			trAr += resISE.getElem(iRow, jCol)*mResid.getElem(iRow, jCol);
		}
	}
	// M[p] crossprodict trace
	double trM = 0.0;
	for (size_t jCol = 0; jCol < M.getNcols(); ++jCol) {
		for (size_t iRow = 0; iRow < M.getNrows(); ++iRow) {
			trM += M.getElem(iRow, jCol)*M.getElem(iRow, jCol);
		}
	}
	trM *= tauP_;

	// now sum to get the log-posterior
	lnP = -0.5*(trResid + trB + trAr + trM);
	return lnP;
}

void MumiLoc::gradient(const vector<double> &theta, vector<double> &grad) const{

}

