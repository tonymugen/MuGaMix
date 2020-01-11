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
MumiLoc::MumiLoc(const vector<double> *yVec, const vector<double> *iSigVec, const vector<Index> *hierInd, const double &tau) : Model(), hierInd_{hierInd}, tau0_{tau}, iSigTheta_{iSigVec} {
	const size_t n = (*hierInd_)[0].size();
	if (yVec->size()%n) {
		throw string("ERROR: Y dimensions not compatible with the number of data points implied by the replicate factor");
	}
	const size_t d = yVec->size()/n;
	Y_     = MatrixViewConst(yVec, 0, n, d);

	vLx_.resize(2*d*d, 0.0);
	Le_ = MatrixView(&vLx_, 0, d, d);
	La_ = MatrixView(&vLx_, d*d, d, d);
	for (size_t k = 0; k < d; k++) {
		Le_.setElem(k, k, 1.0);
		La_.setElem(k, k, 1.0);
	}
	size_t trLen = d*(d-1)/2;
	fTeInd_      = trLen;
	fLaInd_      = trLen + d;
	fTaInd_      = fLaInd_ + trLen;
	fTpInd_      = fTaInd_ + d;
}

MumiLoc::MumiLoc(MumiLoc &&in) {
	if (this != &in) {
		Y_       = move(in.Y_);
		tau0_    = in.tau0_;
		hierInd_ = in.hierInd_;
		Le_      = move(in.Le_);
		La_      = move(in.La_);
		vLx_     = move(in.vLx_);
		fTeInd_  = in.fTeInd_;
		fLaInd_  = in.fLaInd_;
		fTaInd_  = in.fTaInd_;

		in.hierInd_   = nullptr;
		in.iSigTheta_ = nullptr;
	}
}


MumiLoc& MumiLoc::operator=(MumiLoc &&in){
	if (this != &in) {
		Y_       = move(in.Y_);
		tau0_    = in.tau0_;
		hierInd_ = in.hierInd_;
		Le_      = move(in.Le_);
		La_      = move(in.La_);
		vLx_     = move(in.vLx_);
		fTeInd_  = in.fTeInd_;
		fLaInd_  = in.fLaInd_;
		fTaInd_  = in.fTaInd_;

		in.hierInd_   = nullptr;
		in.iSigTheta_ = nullptr;
	}
	return *this;
}

void MumiLoc::expandISvec_() const{
	size_t eInd = 0;                                                      // index of the Le lower triangle in the input vector
	size_t aInd = fLaInd_;                                                // index of the La lower triangle in the input vector
	for (size_t jCol = 0; jCol < Y_.getNcols() - 1; jCol++) {             // the last column is all 0, except the last element = 1.0
		for (size_t iRow = jCol + 1; iRow < Y_.getNcols(); iRow++) {
			Le_.setElem(iRow, jCol, (*iSigTheta_)[eInd]);
			eInd++;
			La_.setElem(iRow, jCol, (*iSigTheta_)[aInd]);
			aInd++;
		}
	}
}

double MumiLoc::logPost(const vector<double> &theta) const{
	// make L matrices
	expandISvec_();
	const size_t Nln  = (*hierInd_)[0].groupNumber();
	const size_t Npop = (*hierInd_)[1].groupNumber();
	const size_t Ydim = Y_.getNrows()*Y_.getNcols();
	MatrixViewConst A(&theta, 0, Nln, Y_.getNcols() );
	MatrixViewConst Mp(&theta, Nln*Y_.getNcols(), Npop, Y_.getNcols() );
	MatrixViewConst mu(&theta, (Nln+Npop)*Y_.getNcols(), 1, Y_.getNcols()); // overall mean

	// Calculate the residual Y - ZA matrix
	vector<double> vResid(Ydim, 0.0);
	MatrixView mResid(&vResid, 0, Y_.getNrows(), Y_.getNcols());
	A.colExpand((*hierInd_)[0], mResid); // ZA
	for (size_t jCol = 0; jCol  < Y_.getNcols(); ++jCol) {
		for (size_t iRow = 0; iRow < Y_.getNrows(); ++ iRow) {
			double diff =  Y_.getElem(iRow, jCol) - mResid.getElem(iRow, jCol);
			mResid.setElem( iRow, jCol, diff); // Y - ZA
		}
	}
	// multiply by L_E
	mResid.trm('l', 'r', false, true, 1.0, Le_);
	// Now calculate the trace of the RL_ET_EL_^TR^T matrix
	double eTrace = 0.0;
	for (size_t jCol = 0; jCol < Y_.getNcols(); jCol++) {
		double dp = 0.0;
		for (size_t iRow = 0; iRow < Y_.getNrows(); iRow++) {
			dp += mResid.getElem(iRow, jCol)*mResid.getElem(iRow, jCol);
		}
		eTrace += exp((*iSigTheta_)[fTeInd_ + jCol])*dp;
	}
	vResid.clear();
	// Clear the Y residuals and re-use for A residuals
	vResid.clear();
	vResid.resize(A.getNrows()*A.getNcols(), 0.0);
	mResid = MatrixView(&vResid, 0, A.getNrows(), A.getNcols());
	Mp.colExpand((*hierInd_)[1], mResid);
	for (size_t jCol = 0; jCol  < A.getNcols(); ++jCol) {
		for (size_t iRow = 0; iRow < A.getNrows(); ++iRow) {
			double diff =  A.getElem(iRow, jCol) - mResid.getElem(iRow, jCol);
			mResid.setElem( iRow, jCol, diff); // A - Z[p]M[p]
		}
	}
	mResid.trm('l', 'r', false, true, 1.0, La_);
	// Now calculate the trace of the RL_AT_AL_^TR^T matrix
	double aTrace = 0.0;
	for (size_t jCol = 0; jCol < A.getNcols(); jCol++) {
		double dp = 0.0;
		for (size_t iRow = 0; iRow < A.getNrows(); iRow++) {
			dp += mResid.getElem(iRow, jCol)*mResid.getElem(iRow, jCol);
		}
		aTrace += exp((*iSigTheta_)[fTaInd_ + jCol])*dp;
	}
	// M[p] crossproduct trace
	double trM = 0.0;
	for (size_t jCol = 0; jCol < Mp.getNcols(); ++jCol) {
		for (size_t iRow = 0; iRow < Mp.getNrows(); ++iRow) {
			double diff = Mp.getElem(iRow, jCol) - mu.getElem(0, jCol);
			trM += diff*exp((*iSigTheta_)[fTpInd_ + jCol])*diff;
		}
	}
	double trP = 0.0;
	for (size_t jCol = 0; jCol < Y_.getNcols(); jCol++) {
		trP += mu.getElem(0, jCol)*mu.getElem(0, jCol);
	}
	trP *= tau0_;
	// now sum to get the log-posterior
	return -0.5*(eTrace + aTrace + trM + trP);
}

void MumiLoc::gradient(const vector<double> &theta, vector<double> &grad) const{
	expandISvec_();
	const size_t Nln  = (*hierInd_)[0].groupNumber();
	const size_t Npop = (*hierInd_)[1].groupNumber();
	const size_t Ydim = Y_.getNrows()*Y_.getNcols();
	const size_t Adim = Nln*Y_.getNcols();
	MatrixViewConst A(&theta, 0, Nln, Y_.getNcols() );
	MatrixViewConst Mp(&theta, Adim, Npop, Y_.getNcols() );
	MatrixViewConst mu(&theta, Adim + Npop*Y_.getNcols(), 1, Y_.getNcols()); // overall mean

	// Matrix views of the gradient
	MatrixView gA(&grad, 0, Nln, Y_.getNcols() );
	MatrixView gMp(&grad, Adim, Npop, Y_.getNcols() );
	MatrixView gmu(&grad, Adim + Npop*Y_.getNcols(), 1, Y_.getNcols()); // overall mean

	// Calculate the residual Y - ZA matrix
	vector<double> vResid(Ydim, 0.0);
	MatrixView mResid(&vResid, 0, Y_.getNrows(), Y_.getNcols());
	A.colExpand((*hierInd_)[0], mResid); // ZA
	for (size_t jCol = 0; jCol  < Y_.getNcols(); ++jCol) {
		for (size_t iRow = 0; iRow < Y_.getNrows(); ++ iRow) {
			double diff =  Y_.getElem(iRow, jCol) - mResid.getElem(iRow, jCol);
			mResid.setElem( iRow, jCol, diff); // Y - ZA
		}
	}

	// Make precision matrices
	vector<double> vISigE(Y_.getNcols()*Y_.getNcols(), 0.0);
	MatrixView iSigE(&vISigE, 0, Y_.getNcols(), Y_.getNcols());
	// L_ExT_E
	vector<double> Tx;
	for (size_t k = 0; k < Y_.getNcols(); k++) {
		Tx.push_back(exp((*iSigTheta_)[fTeInd_ + k]));
	}
	for (size_t jCol = 0; jCol < Y_.getNcols() - 1; jCol++) {
		iSigE.setElem(jCol, jCol, Tx[jCol]);
		for (size_t iRow = jCol + 1; iRow < Y_.getNcols(); iRow++) {
			iSigE.setElem(iRow, jCol, Le_.getElem(iRow, jCol)*Tx[jCol]);
		}
	}
	// last element of the vector is the lower right corner element
	vISigE.back() = Tx.back();
	iSigE.trm('l', 'r', true, true, 1.0, Le_);
	vector<double> vISigA(Y_.getNcols()*Y_.getNcols(), 0.0);
	MatrixView iSigA(&vISigA, 0, Y_.getNcols(), Y_.getNcols());
	for (size_t k = 0; k < Y_.getNcols(); k++) {
		Tx[k] = exp((*iSigTheta_)[fTaInd_ + k]);
	}
	for (size_t jCol = 0; jCol < Y_.getNcols() - 1; jCol++) {
		iSigA.setElem(jCol, jCol, Tx[jCol]);
		for (size_t iRow = jCol + 1; iRow < Y_.getNcols(); iRow++) {
			iSigA.setElem(iRow, jCol, La_.getElem(iRow, jCol)*Tx[jCol]);
		}
	}
	vISigA.back() = Tx.back();
	iSigA.trm('l', 'r', true, true, 1.0, La_);
	// (Y - ZA)Sig[E]^-1
	vector<double> vResISE(Ydim, 0.0);
	MatrixView mResISE(&vResISE, 0, Y_.getNrows(), Y_.getNcols());
	mResid.symm('l', 'r', 1.0, iSigE, 0.0, mResISE);

	// Calculate the residual A - Z[p]M[p]
	vector<double> vAMresid(Adim, 0.0);
	MatrixView mAMresid(&vAMresid, 0, A.getNrows(), A.getNcols());
	Mp.colExpand((*hierInd_)[1], mAMresid); // Z[p]M[p]
	for (size_t jCol = 0; jCol  < A.getNcols(); ++jCol) {
		for (size_t iRow = 0; iRow < A.getNrows(); ++iRow) {
			double diff =  A.getElem(iRow, jCol) - mAMresid.getElem(iRow, jCol);
			mAMresid.setElem(iRow, jCol, diff); // A - Z[p]M[p]
		}
	}
	// (A-Z[p]M[p])Sig^{-1}[A]
	vector<double> vAMresISA(Adim, 0.0);
	MatrixView mAMresISA(&vAMresISA, 0, A.getNrows(), A.getNcols());
	mAMresid.symm('l', 'r', 1.0, iSigA, 0.0, mAMresISA);

	// Z^T(Y - ZA - XB)Sig[E]^-1 store in gA
	mResISE.colSums((*hierInd_)[0], gA);
	// A partial derivatives
	for (size_t jCol = 0; jCol < A.getNcols(); ++jCol) {
		for (size_t iRow = 0; iRow < A.getNrows(); ++iRow) {
			double diff =  gA.getElem(iRow, jCol) - mAMresISA.getElem(iRow, jCol);
			gA.setElem(iRow, jCol, diff); // Z^T(Y - ZA - XB)Sig[E]^-1 - (A-Z[p]M[p])Sig^{-1}[A]
		}
	}
	// Z[p]^T(A - Z[p]M[p])Sig[A]^-1
	mAMresISA.colSums((*hierInd_)[1], gMp);
	// make tau_p
	vector<double> tauP;
	for (size_t k = fTpInd_; k < iSigTheta_->size(); k++) {
		tauP.push_back(exp( (*iSigTheta_)[k] ));
	}
	// M[p] partial derivatives
	for (size_t jCol = 0; jCol < Mp.getNcols(); ++jCol) {
		for (size_t iRow = 0; iRow < Mp.getNrows(); ++iRow) {
			double diff =  gMp.getElem(iRow, jCol) - (Mp.getElem(iRow, jCol) - mu.getElem(0, jCol))*tauP[jCol];
			gMp.setElem( iRow, jCol, diff); // Z[p]^T(A - Z[p]M[p])Sig[A]^-1 - (M[p] - mu)tau[p]
		}
	}
	// mu partial derivatives
	for (size_t jCol = 0; jCol < Mp.getNcols(); jCol++) {
		double colSum = 0.0;
		for (size_t iRow = 0; iRow < Mp.getNrows(); iRow++) {
			colSum += Mp.getElem(iRow, jCol);
		}
		colSum -= static_cast<double>(Npop)*mu.getElem(0, jCol); // Z[m]^T(M_p - mu)
		colSum *= tauP[jCol];
		colSum -= tau0_*mu.getElem(0, jCol);
		gmu.setElem(0, jCol, colSum);
	}
}

// MumiISig methods
MumiISig::MumiISig(const vector<double> *yVec, const vector<double> *vTheta, const vector<Index> *hierInd, const double &nu0, const double &invAsq) : Model(), hierInd_{hierInd}, nu0_{nu0}, invAsq_{invAsq} {
	// TODO: delete the tests after code testing; they will all be done by the wrapping class
	const size_t N = (*hierInd_)[0].size(); // first index is data to lines
	if (yVec->size()%N) {
		throw string("MumiISig constructor ERROR: vectorized data length not divisible by number of data points");
	}
	const size_t d = yVec->size()/N;
	if (vTheta->size()%d) {
		throw string("MumiISig constructor ERROR: vectorized parameter set length not divisible by number of traits");
	}
	Y_ = MatrixViewConst(yVec, 0, N, d);
	const size_t Nln = (*hierInd_)[0].groupNumber();
	if ((*hierInd_)[1].size() != Nln) {
		throw string("MumiISig constructor ERROR: number of elements in the population factor not the same as the number of lines in the line factor");
	}
	const size_t Npop = (*hierInd_)[1].groupNumber();
	A_  = MatrixViewConst(vTheta, 0, Nln, d);
	Mp_ = MatrixViewConst(vTheta, Nln*d, Npop, d);
	mu_ = MatrixViewConst(vTheta, (Nln+Npop)*d, 1, d);
	vLx_.resize(2*d*d, 0.0);
	Le_ = MatrixView(&vLx_, 0, d, d);
	La_ = MatrixView(&vLx_, d*d, d, d);
	for (size_t k = 0; k < d; k++) {
		Le_.setElem(k, k, 1.0);
		La_.setElem(k, k, 1.0);
	}
	size_t trLen = d*(d-1)/2;
	fTeInd_      = trLen;
	fLaInd_      = trLen + d;
	fTaInd_      = fLaInd_ + trLen;
	fTpInd_      = fTaInd_ + d;
	nxnd_        = nu0_*(nu0_ + 2.0*static_cast<double>(d));
	Nnd_         = static_cast<double>(Y_.getNrows()) + nu0_ + 2.0*static_cast<double>(d);
	NAnd_        = static_cast<double>(Nln) + nu0_ + 2.0*static_cast<double>(d);
	NPnd_        = static_cast<double>(Npop) + nu0_ + 2.0*static_cast<double>(d);
};

MumiISig::MumiISig(MumiISig &&in) {
	if (this != &in) {
		hierInd_ = in.hierInd_;
		nu0_     = in.nu0_;
		invAsq_  = in.invAsq_;
		Y_       = move(in.Y_);
		A_       = move(in.A_);
		B_       = move(in.B_);
		Mp_      = move(in.Mp_);
		mu_      = move(in.mu_);
		vLx_     = move(in.vLx_);
		Le_      = MatrixView(&vLx_, 0, Y_.getNcols(), Y_.getNcols());
		La_      = MatrixView(&vLx_, Y_.getNcols()*Y_.getNcols(), Y_.getNcols(), Y_.getNcols());
		fTeInd_  = in.fTeInd_;
		fLaInd_  = in.fLaInd_;
		fTaInd_  = in.fTaInd_;
		nxnd_    = in.nxnd_;
		Nnd_     = in.Nnd_;
		NAnd_    = in.NAnd_;
		NPnd_    = in.NPnd_;

		in.hierInd_ = nullptr;
	}
};

MumiISig& MumiISig::operator=(MumiISig &&in){
	if (this != &in) {
		hierInd_ = in.hierInd_;
		nu0_     = in.nu0_;
		invAsq_  = in.invAsq_;
		Y_       = move(in.Y_);
		A_       = move(in.A_);
		B_       = move(in.B_);
		Mp_      = move(in.Mp_);
		mu_      = move(in.mu_);
		vLx_     = move(in.vLx_);
		Le_      = MatrixView(&vLx_, 0, Y_.getNcols(), Y_.getNcols());
		La_      = MatrixView(&vLx_, Y_.getNcols()*Y_.getNcols(), Y_.getNcols(), Y_.getNcols());
		fTeInd_  = in.fTeInd_;
		fLaInd_  = in.fLaInd_;
		fTaInd_  = in.fTaInd_;
		fTpInd_  = in.fTpInd_;
		nxnd_    = in.nxnd_;
		Nnd_     = in.Nnd_;
		NPnd_    = in.NPnd_;

		in.hierInd_ = nullptr;
	}
	return *this;
}

void MumiISig::expandISvec_(const vector<double> &viSig) const{
	size_t eInd = 0;                                                      // index of the Le lower triangle in the input vector
	size_t aInd = fLaInd_;                                                // index of the La lower triangle in the input vector
	for (size_t jCol = 0; jCol < Y_.getNcols() - 1; jCol++) {             // the last column is all 0, except the last element = 1.0
		for (size_t iRow = jCol + 1; iRow < Y_.getNcols(); iRow++) {
			Le_.setElem(iRow, jCol, viSig[eInd]);
			eInd++;
			La_.setElem(iRow, jCol, viSig[aInd]);
			aInd++;
		}
	}
}

double MumiISig::logPost(const vector<double> &viSig) const{
	// expand the element vector to make the L matrices
	expandISvec_(viSig);
	// Calculate the Y residuals
	vector<double> vResid(Y_.getNcols()*Y_.getNrows(), 0.0);
	MatrixView mResid(&vResid, 0, Y_.getNrows(), Y_.getNcols());
	A_.colExpand((*hierInd_)[0], mResid); // ZA
	for (size_t jCol = 0; jCol  < Y_.getNcols(); ++jCol) {
		for (size_t iRow = 0; iRow < Y_.getNrows(); ++ iRow) {
			double diff = Y_.getElem(iRow, jCol) - mResid.getElem(iRow, jCol);
			mResid.setElem( iRow, jCol, diff); // Y - ZA
		}
	}
	// multiply by L_E
	mResid.trm('l', 'r', false, true, 1.0, Le_);
	// Now calculate the trace of the RL_ET_EL_^TR^T matrix
	double eTrace = 0.0;
	for (size_t jCol = 0; jCol < Y_.getNcols(); jCol++) {
		double dp = 0.0;
		for (size_t iRow = 0; iRow < Y_.getNrows(); iRow++) {
			dp += mResid.getElem(iRow, jCol)*mResid.getElem(iRow, jCol);
		}
		eTrace += exp(viSig[fTeInd_ + jCol])*dp;
	}

	// Clear the Y residuals and re-use for A residuals
	vResid.clear();
	vResid.resize(A_.getNrows()*A_.getNcols(), 0.0);
	mResid = MatrixView(&vResid, 0, A_.getNrows(), A_.getNcols());
	Mp_.colExpand((*hierInd_)[1], mResid);
	for (size_t jCol = 0; jCol  < A_.getNcols(); ++jCol) {
		for (size_t iRow = 0; iRow < A_.getNrows(); ++iRow) {
			double diff =  A_.getElem(iRow, jCol) - mResid.getElem(iRow, jCol);
			mResid.setElem( iRow, jCol, diff); // A - Z[p]M[p]
		}
	}
	mResid.trm('l', 'r', false, true, 1.0, La_);
	// Now calculate the trace of the RL_AT_AL_^TR^T matrix
	double aTrace = 0.0;
	for (size_t jCol = 0; jCol < A_.getNcols(); jCol++) {
		double dp = 0.0;
		for (size_t iRow = 0; iRow < A_.getNrows(); iRow++) {
			dp += mResid.getElem(iRow, jCol)*mResid.getElem(iRow, jCol);
		}
		aTrace += exp(viSig[fTaInd_ + jCol])*dp;
	}
	// Mp trace
	double trM = 0.0;
	for (size_t jCol = 0; jCol < Mp_.getNcols(); ++jCol) {
		for (size_t iRow = 0; iRow < Mp_.getNrows(); ++iRow) {
			double diff = Mp_.getElem(iRow, jCol) - mu_.getElem(0, jCol);
			trM += diff*exp(viSig[fTpInd_ + jCol])*diff;
		}
	}
	// Sum of log-determinants
	double ldetSumE = 0.0;
	double ldetSumA = 0.0;
	double ldetSumP = 0.0;
	for (size_t k = 0; k < Y_.getNcols(); k++) {
		ldetSumE += viSig[fTeInd_ + k];
		ldetSumA += viSig[fTaInd_ + k];
		ldetSumP += viSig[fTpInd_ + k];
	}
	ldetSumE *= static_cast<double>(Y_.getNrows() + 2*Y_.getNcols()) + nu0_;
	ldetSumA *= static_cast<double>(A_.getNrows() + 2*A_.getNcols()) + nu0_;
	ldetSumP *= static_cast<double>(Mp_.getNrows() + 2*Mp_.getNcols()) + nu0_;
	// Calculate the prior components; k and m are as in the derivation document; doing the L_E and L_A in one pass
	// first element has just the diagonal
	double pTrace = log(nu0_*exp(viSig[fTeInd_]) + invAsq_) + log(nu0_*exp(viSig[fTaInd_]) + invAsq_);
	for (size_t k = 1; k < Le_.getNcols(); k++) { // k starts from the second element (k=1)
		double sE = 0.0;
		double sA = 0.0;
		for (size_t m = 0; m <= k - 1; m++) { // the <= is intentional; excluding only m = k
			sE += exp(viSig[fTeInd_ + m])*Le_.getElem(k, m)*Le_.getElem(k, m);
			sA += exp(viSig[fTaInd_ + m])*La_.getElem(k, m)*La_.getElem(k, m);
		}
		sE += exp(viSig[fTeInd_ + k]);
		sA += exp(viSig[fTaInd_ + k]);
		pTrace += log(nu0_*sE + invAsq_) + log(nu0_*sA + invAsq_) + log(nu0_*exp(viSig[fTpInd_ + k]) + invAsq_);
	}
	pTrace *= nu0_ + 2.0*static_cast<double>(Y_.getNcols());
	return -0.5*(eTrace + aTrace + trM - ldetSumE - ldetSumA - ldetSumP + pTrace);
}

void MumiISig::gradient(const vector<double> &viSig, vector<double> &grad) const{
	// expand the element vector to make the L matrices
	expandISvec_(viSig);
	// Calculate the Y residuals
	vector<double> vResid(Y_.getNcols()*Y_.getNrows(), 0.0);
	MatrixView mResid(&vResid, 0, Y_.getNrows(), Y_.getNcols());
	A_.colExpand((*hierInd_)[0], mResid); // ZA
	for (size_t jCol = 0; jCol  < Y_.getNcols(); ++jCol) {
		for (size_t iRow = 0; iRow < Y_.getNrows(); ++ iRow) {
			double diff =  Y_.getElem(iRow, jCol) - mResid.getElem(iRow, jCol);
			mResid.setElem( iRow, jCol, diff); // Y - ZA
		}
	}
	vector<double> vRtR(Y_.getNcols()*Y_.getNcols(), 0.0);
	MatrixView mRtR(&vRtR, 0, Y_.getNcols(), Y_.getNcols());
	mResid.syrk('l', 1.0, 0.0, mRtR);
	vector<double> vRtRLT(Y_.getNcols()*Y_.getNcols(), 0.0);
	MatrixView mRtRLT(&vRtRLT, 0, Y_.getNcols(), Y_.getNcols());
	Le_.symm('l', 'l', 1.0, mRtR, 0.0, mRtRLT); // R^TRL_E; R = Y - ZA
	// make a vector of T_X (provided values are on the log scale; will use for T_E and T_A)
	vector<double> Tx;
	for (size_t k = 0; k < Y_.getNcols(); k++) {
		Tx.push_back(exp(viSig[fTeInd_ + k]));
	}
	// mutiply by T_E (the whole matrix because I will need it for left-multiplication later)
	for (size_t jCol = 0; jCol < Y_.getNcols(); jCol++) {
		for (size_t iRow = 0; iRow < Y_.getNcols(); iRow++) {
			double prod = mRtRLT.getElem(iRow, jCol)*Tx[jCol];
			mRtRLT.setElem(iRow, jCol, prod);
		}
	}
	// construct the weighted L_E
	// start with unweighted values because they can be used in weight calculations
	vector<double> vechLwX;                                     // vech(L^w_X)
	vector<double> weights(Y_.getNcols(), 0.0);                 // will become a d-vector of weights (each element corresponding to a row of L_X; the first element is weighted T_E[1,1])
	for (size_t jCol = 0; jCol < Y_.getNcols() - 1; jCol++) {   // nothing to be done for the last column (it only has a diagonal element)
		for (size_t iRow = jCol + 1; iRow < Y_.getNcols(); iRow++) {
			double prod1 = Tx[jCol]*Le_.getElem(iRow, jCol);
			vechLwX.push_back(prod1);
			weights[iRow] += prod1*Le_.getElem(iRow, jCol); // unweighted for now
		}
	}
	for (size_t k = 0; k < Y_.getNcols(); k++) {
		weights[k] = nu0_*(weights[k] + Tx[k]) + invAsq_;
	}
	size_t vechInd = 0;
	for (size_t jCol = 0; jCol < Y_.getNcols() - 1; jCol++) {
		for (size_t iRow = jCol + 1; iRow < Y_.getNcols(); iRow++) {
			vechLwX[vechInd] = vechLwX[vechInd]/weights[iRow];
			vechInd++;
		}
	}
	// add the lower triangles and store the results in the gradient vector
	vechInd = 0;
	for (size_t jCol = 0; jCol < Y_.getNcols() - 1; jCol++) {
		for (size_t iRow = jCol + 1; iRow < Y_.getNcols(); iRow++) {
			grad[vechInd] = -mRtRLT.getElem(iRow, jCol) - nxnd_*vechLwX[vechInd];
			vechInd++;
		}
	}
	// The T_E gradient
	// Starting with the first matrix: mRtRLT becomes L_E^TR^TRL_ET_E
	mRtRLT.trm('l', 'l', true, true, 1.0, Le_);
	// now sum everything and store the result in the gradient vector
	for (size_t k = 0; k < Y_.getNcols(); k++) {
		grad[fTeInd_ + k] = 0.5*(Nnd_ - mRtRLT.getElem(k, k) - nxnd_*Tx[k]/weights[k]);
	}
	//
	// L_A and T_A next
	//
	// Start with caclulating the residual, replacing the old one
	vResid.resize(A_.getNcols()*A_.getNrows(), 0.0);
	mResid = MatrixView(&vResid, 0, A_.getNrows(), A_.getNcols());
	Mp_.colExpand((*hierInd_)[1], mResid); // Z_pM_p
	for (size_t jCol = 0; jCol  < A_.getNcols(); ++jCol) {
		for (size_t iRow = 0; iRow < A_.getNrows(); ++ iRow) {
			double diff =  A_.getElem(iRow, jCol) - mResid.getElem(iRow, jCol);
			mResid.setElem( iRow, jCol, diff); // A - Z_pM_p
		}
	}
	// Calculate the crossproduct and multiply by L_A
	mResid.syrk('l', 1.0, 0.0, mRtR);
	La_.symm('l', 'l', 1.0, mRtR, 0.0, mRtRLT); // R^TRL_A; R = A - Z_pM_p
	for (size_t k = 0; k < A_.getNcols(); k++) {
		Tx[k] = exp(viSig[fTaInd_ + k]);
	}
	// mutiply by T_A (the whole matrix because I will need it for left-multiplication later)
	for (size_t jCol = 0; jCol < A_.getNcols(); jCol++) {
		for (size_t iRow = 0; iRow < A_.getNcols(); iRow++) {
			double prod = mRtRLT.getElem(iRow, jCol)*Tx[jCol];
			mRtRLT.setElem(iRow, jCol, prod);
		}
	}
	weights.assign(weights.size(), 0.0);
	vechInd = 0;
	for (size_t jCol = 0; jCol < A_.getNcols() - 1; jCol++) {   // nothing to be done for the last column (it only has a diagonal element)
		for (size_t iRow = jCol + 1; iRow < A_.getNcols(); iRow++) {
			double prod1 = Tx[jCol]*La_.getElem(iRow, jCol);
			vechLwX[vechInd] = prod1;
			weights[iRow] += prod1*La_.getElem(iRow, jCol); // unweighted for now
			vechInd++;
		}
	}
	for (size_t k = 0; k < A_.getNcols(); k++) {
		weights[k] = nu0_*(weights[k] + Tx[k]) + invAsq_;
	}
	vechInd = 0;
	for (size_t jCol = 0; jCol < Y_.getNcols() - 1; jCol++) {
		for (size_t iRow = jCol + 1; iRow < Y_.getNcols(); iRow++) {
			vechLwX[vechInd] = vechLwX[vechInd]/weights[iRow];
			vechInd++;
		}
	}
	// add the lower triangles and store the results in the gradient vector
	vechInd = 0;
	for (size_t jCol = 0; jCol < Y_.getNcols() - 1; jCol++) {
		for (size_t iRow = jCol + 1; iRow < Y_.getNcols(); iRow++) {
			grad[fLaInd_+vechInd] = -mRtRLT.getElem(iRow, jCol) - nxnd_*vechLwX[vechInd];
			vechInd++;
		}
	}
	// The T_A gradient
	// Starting with the first matrix: mRtRLT becomes L_A^TR^TRL_AT_A
	mRtRLT.trm('l', 'l', true, true, 1.0, La_);
	// now sum everything and store the result in the gradient vector
	for (size_t k = 0; k < A_.getNcols(); k++) {
		grad[fTaInd_ + k] = 0.5*(NAnd_ - mRtRLT.getElem(k, k) - nxnd_*Tx[k]/weights[k]);
	}
	// The T_P gradient
	// Start with caclulating the residual, replacing the old one
	vResid.clear();
	for (size_t jCol = 0; jCol < Mp_.getNcols(); jCol++) {
		for (size_t iRow = 0; iRow < Mp_.getNrows(); iRow++) {
			vResid.push_back(Mp_.getElem(iRow, jCol) - mu_.getElem(0, jCol));
		}
	}
	mResid = MatrixView(&vResid, 0, Mp_.getNrows(), Mp_.getNcols());
	mResid.syrk('l', 1.0, 0.0, mRtR);
	for (size_t k = 0; k < A_.getNcols(); k++) {
		Tx[k] = exp(viSig[fTpInd_ + k]);
	}
	for (size_t k = 0; k < Mp_.getNcols(); k++) {
		weights[k] = nu0_*Tx[k] + invAsq_;
	}
	// now sum everything and store the result in the gradient vector
	for (size_t k = 0; k < Mp_.getNcols(); k++) {
		grad[fTpInd_ + k] = 0.5*(NPnd_ - mRtR.getElem(k, k)*Tx[k] - nxnd_*Tx[k]/weights[k]);
	}
}

// WrapMM methods
WrapMMM::WrapMMM(const vector<double> &vY, const vector<size_t> &y2line, const uint32_t &Npop, const double &alphaPr, const double &tau0, const double &nu0, const double &invAsq): vY_{vY} {
	hierInd_.push_back(Index(y2line));
	// Set up population mixture vectors
	alpha_.resize(Npop, alphaPr);
	pi_.resize(Npop, 0.0);
	Dmn_.resize(Npop, 0.0);
	// Initialize the line to population index
	for (size_t iA = 0; iA < hierInd_[0].groupNumber(); iA++) {
		z_.push_back(rng_.sampleInt(Npop)); // the max value is not included
	}
	// deterministic assignment to true populations for now to test
	/*
	for (size_t k = 0; k < Npop; k++) {
		for (size_t i = 0; i < hierInd_[0].groupNumber()/Npop; i++) {
			z_.push_back(k);
		}
	}
	*/
	hierInd_.push_back(Index(Npop));
	hierInd_.back().update(z_);
	updatePi_();
	// TODO: take out after debugging is done
	if (hierInd_[0].groupNumber() != hierInd_[1].size()) {
		throw string("WrapMMM constructor ERROR: the line and population hierarchical indexes do not match");
	}
	models_.push_back( new MumiLoc(&vY_, &vISig_, &hierInd_, tau0) );
	models_.push_back( new MumiISig(&vY_, &vTheta_, &hierInd_, nu0, invAsq) );
	const size_t N = hierInd_[0].size();
	if (vY.size()%N) {
		throw string("WrapMMM constructor ERROR: length of response vector not divisible by data point number");
	}
	const size_t d     = vY.size()/N;
	const size_t Nln   = hierInd_[0].groupNumber();
	const size_t Adim  = Nln*d;
	const size_t Mpdim = Npop*d;

	// Calculate starting values for theta
	MatrixViewConst Y(&vY_, 0, N, d);

	vTheta_.resize(Adim + Mpdim + d, 0.0);
	vPz_.resize(Npop*Nln, 0.0);
	A_  = MatrixView(&vTheta_, 0, Nln, d);
	Mp_ = MatrixView(&vTheta_, Adim, Npop, d);
	MatrixView mu(&vTheta_, Adim+Mpdim, 1, d);
	Pz_ = MatrixView(&vPz_, 0, Npop, Nln);

	Y.colMeans(hierInd_[0], A_);    //  means to get A starting values
	A_.colMeans(hierInd_[1], Mp_);    // A means to get population mean starting values
	vector<double> tmpMu;
	Mp_.colMeans(tmpMu);
	for (size_t k = 0; k < d; k++) {
		mu.setElem(0, k, tmpMu[k]);
	}

	// Calculate starting precision matrix values; do that before adding noise to theta
	//
	size_t trLen     = d*(d-1)/2;
	fLaInd_          = trLen + d;
	fTaInd_          = fLaInd_ + trLen;
	const double n   = 1.0/static_cast<double>(N-1);
	const double nLN = 1.0/static_cast<double>(Nln-1);
	const double nP  = static_cast<double>(Npop-1); // not reciprocal on purpose
	vLa_.resize(d*d, 0.0);
	La_ = MatrixView(&vLa_, 0, d, d);
	vector<double> vSig(d*d, 0.0);
	MatrixView Sig(&vSig, 0, d, d);

	// Y residual
	vector<double> vZA(N*d, 0.0);
	MatrixView ZA(&vZA, 0, N, d);
	A_.colExpand(hierInd_[0], ZA);
	for (size_t jCol = 0; jCol < d; jCol++) {
		for (size_t iRow = 0; iRow < N; iRow++) {
			double diff = Y.getElem(iRow, jCol) - ZA.getElem(iRow, jCol);
			ZA.setElem(iRow, jCol, diff); // ZA now Y - ZA
		}
	}
	ZA.syrk('l', n, 0.0, Sig); // making covariances in one step
	// add a small value to the diagonal to make sure the matrix is non-singular
	for (size_t k = 0; k < d; k++) {
		double diag = Sig.getElem(k, k);
		Sig.setElem(k, k, diag + 1e-4);
	}
	Sig.chol();
	Sig.cholInv();

	// save the scaled precision matrix lower triangle and log-diagonals to the precision parameter vector
	vector<double> sqrT;
	for (size_t k = 0; k < d; k++) {
		sqrT.push_back(sqrt(Sig.getElem(k, k)));
	}
	for (size_t jCol = 0; jCol < d-1; jCol++) {
		for (size_t iRow = jCol+1; iRow < d; iRow++) {
			vISig_.push_back( Sig.getElem(iRow, jCol)/(sqrT[iRow]*sqrT[jCol]) );
		}
	}
	for (size_t k = 0; k < d; k++) {
		vISig_.push_back( log(Sig.getElem(k, k)) );
	}
	// A precision matrix
	vZA.resize(Nln*d);
	MatrixView ZpMp(&vZA, 0, Nln, d);
	Mp_.colExpand(hierInd_[1], ZpMp);
	for (size_t jCol = 0; jCol < d ; jCol++) {
		for (size_t iRow = 0; iRow < Nln ; iRow++) {
			double diff = A_.getElem(iRow, jCol) - ZpMp.getElem(iRow, jCol);
			ZpMp.setElem(iRow, jCol, diff); // A - ZM is now in ZM
		}
	}
	ZpMp.syrk('l', nLN, 0.0, Sig);
	// add a small value to the diagonal again
	for (size_t k = 0; k < d ; k++) {
		double diag = Sig.getElem(k, k) + 1e-4;
		Sig.setElem(k, k, diag);
	}
	Sig.chol();
	Sig.cholInv();
	for (size_t k = 0; k < d; k++) {
		sqrT[k] = sqrt(Sig.getElem(k, k));
	}
	for (size_t jCol = 0; jCol < d-1; jCol++) {
		for (size_t iRow = jCol+1; iRow < d; iRow++) {
			vISig_.push_back( Sig.getElem(iRow, jCol)/(sqrT[iRow]*sqrT[jCol]) );
		}
	}
	for (size_t k = 0; k < d; k++) {
		vISig_.push_back( log(Sig.getElem(k, k)) );
	}
	// tau_p
	for (size_t jCol = 0; jCol < d; jCol++) {
		double sSq = 0.0;
		for (size_t iRow = 0; iRow < Mp_.getNrows(); iRow++) {
			double diff = Mp_.getElem(iRow, jCol) - mu.getElem(0, jCol);
			sSq += diff*diff;
		}
		vISig_.push_back( log(nP/sSq) );
	}
	expandLa_();
	vAresid_.resize(Adim, 0.0);
	Aresid_ = MatrixView(&vAresid_, 0, Nln, d);
	// add noise
	for (auto &t : vTheta_) {
		t += 0.5*rng_.rnorm();
	}
	for (auto &s : vISig_) {
		s += 0.5*rng_.rnorm();
	}
	sampler_.push_back( new SamplerNUTS(models_[0], &vTheta_) );
	sampler_.push_back( new SamplerNUTS(models_[1], &vISig_) );
}

WrapMMM::~WrapMMM(){
	for (auto &m : models_) {
		delete m;
	}
	for (auto &s : sampler_) {
		delete s;
	}
}

void WrapMMM::updatePi_(){
	for (size_t k = 0; k < Dmn_.size(); k++) {
		Dmn_[k] = alpha_[k] + static_cast<double>( hierInd_.back().groupSize(k) ); // the cline (accession) to population index always the last one in the vector
	}
	rng_.rdirichlet(Dmn_, pi_);
}

void WrapMMM::expandLa_(){
	vector <double> Ta;
	for (size_t k = fTaInd_; k < fTaInd_ + La_.getNcols(); k++) {
		Ta.push_back( exp(0.5*vISig_[k]) );
	}
	size_t aInd = fLaInd_;                                             // index of the La lower triangle in the input vector
	for (size_t jCol = 0; jCol < La_.getNcols(); jCol++) {             // the last column is all 0, except the last element = Ta[d]
		La_.setElem(jCol, jCol, Ta[jCol]);
		for (size_t iRow = jCol + 1; iRow < La_.getNcols(); iRow++) {
			La_.setElem(iRow, jCol, vISig_[aInd]*Ta[jCol]);
			aInd++;
		}
	}
}

void WrapMMM::updatePz_(){
	for (size_t p = 0; p < Mp_.getNrows(); p++) {             // for each population
		for (size_t jCol = 0; jCol < A_.getNcols(); jCol++) {
			for (size_t iRow = 0; iRow < A_.getNrows(); iRow++) {
				Aresid_.setElem(iRow, jCol, A_.getElem(iRow, jCol) - Mp_.getElem(p, jCol)); // will iterate through populations first
			}
		}
		Aresid_.trm('l', 'r', false, false, 1.0, La_); // half-kernel
		for (size_t iA = 0; iA < Aresid_.getNrows(); iA++) {
			double sSq = 0.0;
			for (size_t jCol = 0; jCol < Aresid_.getNcols(); jCol++) {
				sSq += Aresid_.getElem(iA, jCol)*Aresid_.getElem(iA, jCol);
			}
			Pz_.setElem(p, iA, pi_[p]*exp(-0.5*sSq));
		}
	}
	// normalize the scores to create p[ij]
	for (size_t jCol = 0; jCol < Pz_.getNcols(); jCol++) {
		double sum = 0.0;
		for (size_t iPop = 0; iPop < Pz_.getNrows(); iPop++) {
			sum += Pz_.getElem(iPop, jCol);
		}
		for (size_t iPop = 0; iPop < Pz_.getNrows(); iPop++) {
			double div = Pz_.getElem(iPop, jCol)/sum;
			Pz_.setElem(iPop, jCol, div);
		}
	}
	// assign z_ values
	for (size_t iA = 0; iA < Aresid_.getNrows(); iA++) {
		double    u = rng_.runif();
		double cumP = 0.0;
		for (size_t p = 0; p < Mp_.getNrows(); p++) {
			cumP += Pz_.getElem(p, iA);
			if (u <= cumP) {
				z_[iA] = p;
				break;
			}
		}
	}
	hierInd_.back().update(z_);
	if ( hierInd_.back().groupNumber() != 3 ) {
		throw string("Lost a population");
	}
}

void WrapMMM::runSampler(const uint32_t &Nadapt, const uint32_t &Nsample, vector<double> &thetaChain, vector<double> &piChain, vector<uint32_t> &treeLen){
	treeLen.clear();
	for (uint32_t a = 0; a < Nadapt; a++) {
		for (auto &s : sampler_) {
			treeLen.push_back(s->adapt());
		}
		updatePi_();
		updatePz_();
	}
	thetaChain.clear();
	piChain.clear();
	for (uint32_t b = 0; b < Nsample; b++) {
		for (auto &s : sampler_) {
			s->update();
		}
		for (auto &t : vTheta_) {
			thetaChain.push_back(t);
		}
		for (auto &p : vISig_) {
			thetaChain.push_back(p);
		}
		updatePi_();
		updatePz_();
		for (auto &p : vPz_) {
			piChain.push_back(p);
		}
	}
}


