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
	MatrixViewConst A(&theta, 0, Nln, Y_.getNcols() );
	MatrixViewConst B(&theta, Nln*Y_.getNcols(), X_.getNcols(), Y_.getNcols());
	MatrixViewConst M(&theta, (Nln + X_.getNcols())*Y_.getNcols(), Npop, Y_.getNcols() );

	// Calculate the residual Y - XB - ZA matrix
	vector<double> vResid(Ydim, 0.0);
	MatrixView mResid(&vResid, 0, Y_.getNrows(), Y_.getNcols());
	A.rowExpand((*hierInd_)[0], mResid); // ZA
	for (size_t jCol = 0; jCol  < Y_.getNcols(); ++jCol) {
		for (size_t iRow = 0; iRow < Y_.getNrows(); ++ iRow) {
			mResid.setElem( iRow, jCol, Y_.getElem(iRow, jCol) - mResid.getElem(iRow, jCol) ); // Y - ZA
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
		for (size_t iRow = 0; iRow < A.getNrows(); ++iRow) {
			mResid.setElem( iRow, jCol, A.getElem(iRow, jCol) - mResid.getElem(iRow, jCol) ); // A - Z[p]M[p]
		}
	}
	mResid.symm('u', 'r', 1.0, ISigA_, 0.0, resISE); // (A-Z[p]M[p])Sig^{-1}[A]
	double trAr = 0.0;
	for (size_t jCol = 0; jCol < A.getNcols(); ++jCol) {
		for (size_t iRow = 0; iRow < A.getNrows(); ++iRow) {
			trAr += resISE.getElem(iRow, jCol)*mResid.getElem(iRow, jCol);
		}
	}
	// M[p] crossproduct trace
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
	const size_t Nln  = (*hierInd_)[0].groupNumber();
	const size_t Npop = (*hierInd_)[1].groupNumber();
	const size_t Ydim = Y_.getNrows()*Y_.getNcols();
	MatrixViewConst A(&theta, 0, Nln, Y_.getNcols() );
	MatrixViewConst B(&theta, Nln*Y_.getNcols(), X_.getNcols(), Y_.getNcols());
	MatrixViewConst M(&theta, (Nln + X_.getNcols())*Y_.getNcols(), Npop, Y_.getNcols() );
	const size_t Adim = A.getNrows()*A.getNcols();

	// Matrix views of the gradient
	MatrixView gA(&grad, 0, Nln, Y_.getNcols() );
	MatrixView gB(&grad, Nln*Y_.getNcols(), X_.getNcols(), Y_.getNcols());
	MatrixView gM(&grad, (Nln + X_.getNcols())*Y_.getNcols(), Npop, Y_.getNcols() );

	// Calculate the residual Y - XB - ZA matrix
	vector<double> vResid(Ydim, 0.0);
	MatrixView mResid(&vResid, 0, Y_.getNrows(), Y_.getNcols());
	A.rowExpand((*hierInd_)[0], mResid); // ZA
	for (size_t jCol = 0; jCol  < Y_.getNcols(); ++jCol) {
		for (size_t iRow = 0; iRow < Y_.getNrows(); ++ iRow) {
			mResid.setElem( iRow, jCol, Y_.getElem(iRow, jCol) - mResid.getElem(iRow, jCol) ); // Y - ZA
		}
	}
	B.gemm(false, -1.0, X_, false, 1.0, mResid); // Y - ZA - XB

	// (Y - ZA - XB)Sig[E]^-1
	vector<double> vResISE(Ydim, 0.0);
	MatrixView mResISE(&vResISE, 0, Y_.getNrows(), Y_.getNcols());
	mResid.symm('u', 'r', 1.0, ISigE_, 0.0, mResISE);

	// Calculate the residual A - Z[p]M[p]
	vector<double> vAMresid(Adim, 0.0);
	MatrixView mAMresid(&vAMresid, 0, A.getNrows(), A.getNcols());
	M.rowExpand((*hierInd_)[1], mAMresid); // Z[p]M[p]
	for (size_t jCol = 0; jCol  < A.getNcols(); ++jCol) {
		for (size_t iRow = 0; iRow < A.getNrows(); ++iRow) {
			mAMresid.setElem( iRow, jCol, A.getElem(iRow, jCol) - mAMresid.getElem(iRow, jCol) ); // A - Z[p]M[p]
		}
	}
	// (A-Z[p]M[p])Sig^{-1}[A]
	vector<double> vAMresISA(Adim, 0.0);
	MatrixView mAMresISA(&vAMresISA, 0, A.getNrows(), A.getNcols());
	mAMresid.symm('u', 'r', 1.0, ISigA_, 0.0, mAMresISA);

	// Z^T(Y - ZA - XB)Sig[E]^-1 store in gA
	mResISE.rowCollapse((*hierInd_)[0], gA);
	// A partial derivatives
	for (size_t jCol = 0; jCol < A.getNcols(); ++jCol) {
		for (size_t iRow = 0; iRow < A.getNrows(); ++iRow) {
			gA.setElem( iRow, jCol, gA.getElem(iRow, jCol) - mAMresISA.getElem(iRow, jCol) ); // Z^T(Y - ZA - XB)Sig[E]^-1 - (A-Z[p]M[p])Sig^{-1}[A]
		}
	}

	// X^T(Y - ZA - XB)Sig[E]^-1 store in gB
	mResISE.gemm(true, 1.0, X_, false, 0.0, gB);
	// B partial derivatives
	for (size_t jCol = 0; jCol < B.getNcols(); ++jCol) {
		for (size_t iRow = 0; iRow < B.getNrows(); ++iRow) {
			gB.setElem( iRow, jCol, gB.getElem(iRow, jCol) - tau0_*B.getElem(iRow, jCol) ); // X^T(Y - ZA - XB)Sig[E]^-1 - tau[0]B
		}
	}

	// Z[p]^T(A - Z[p]M[p])Sig[A]^-1
	mAMresISA.rowCollapse((*hierInd_)[1], gM);
	// M[p] partial derivatives
	for (size_t jCol = 0; jCol < M.getNcols(); ++jCol) {
		for (size_t iRow = 0; iRow < M.getNrows(); ++iRow) {
			gM.setElem( iRow, jCol, gM.getElem(iRow, jCol) - tauP_*M.getElem(iRow, jCol) ); // Z[p]^T(A - Z[p]M[p])Sig[A]^-1 - tau[p]M[p]
		}
	}

}

// WrapMM methods
WrapMMM::WrapMMM(const vector<double> &vY, const vector<double> &vX, const vector<size_t> &y2line, const vector<size_t> &ln2pop, const size_t &d, const vector<double> &trueISig, const double &tau0): vY_{vY}, vX_{vX}, vISig_{trueISig} {
	hierInd_.push_back(Index(y2line));
	hierInd_.push_back(Index(ln2pop));
	if (hierInd_[0].groupNumber() != hierInd_[1].size()) {
		throw string("WrapMMM constructor ERROR: the line and population hierarchical indexes do not match");
	}
	models_.push_back( new MumiLoc(&vY_, &vISig_, d, &hierInd_, &vX_, tau0) );
	const size_t N   = vY_.size()/d;
	if (N != vY_.size()/d) {
		throw string("WrapMMM constructor ERROR: the line factor not same length as the data");
	}
	if (vX_.size()%N) {
		throw string("WrapMMM constructor ERROR: vectorized X must contain and integer number of predictors");
	}
	const size_t Nln  = ln2pop.size();
	const size_t Npop = hierInd_[1].groupNumber();
	const size_t Nb   = vX_.size()/N;
	const size_t Adim = Nln*d;
	const size_t Bdim = Nb*d;
	const size_t Mdim = Npop*d;

	// Calculate starting values for theta
	MatrixViewConst Y(&vY_, 0, N, d);
	MatrixViewConst X(&vX_, 0, N, Nb);

	vTheta_.resize(Adim + Bdim + Mdim, 0.0);
	MatrixView A(&vTheta_, 0, Nln, d);
	MatrixView B(&vTheta_, Adim, Nb, d);
	MatrixView M(&vTheta_, Adim+Bdim, Npop, d);

	vector<double> vXtX(Nb*Nb, 0.0);
	MatrixView XtX(&vXtX, 0, Nb, Nb);

	X.syrk('u', 1.0, 0.0, XtX);
	XtX.chol();
	XtX.cholInv();
	vector<double> vXtY(Bdim, 0.0);
	MatrixView XtY(&vXtY, 0, Nb, d);
	Y.gemm(true, 1.0, X, false, 0.0, XtY);
	XtY.symm('u', 'l', 1.0, XtX, 0.0, B);

	vector<double> bResid(vY_);
	MatrixView YmXb(&bResid, 0, N, d); // Y - XB
	B.gemm(false, -1.0, X, false, 1.0, YmXb);
	YmXb.colMeans(hierInd_[0], A); // residual means to get A starting values
	A.colMeans(hierInd_[1], M);    // A means to get population mean starting values
	//for (auto &t : vTheta_) {         // add noise
	//	t += rng_.rnorm();
	//}

	sampler_.push_back( new SamplerNUTS(models_[0], &vTheta_) );
}

WrapMMM::~WrapMMM(){
	for (auto &m : models_) {
		delete m;
	}
	for (auto &s : sampler_) {
		delete s;
	}
}


