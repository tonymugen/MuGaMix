/*
 * Copyright (c) 2020 Anthony J. Greenberg
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

/// Variational inference of Gaussian mixture models
/** \file
 * \author Anthony J. Greenberg
 * \copyright Copyright (c) 2020 Anthony J. Greenberg
 * \version 1.0
 *
 *  Implementation of variational Bayes inference of Gaussian mixture models.
 *
 */

#include <algorithm>
#include <math.h>
#include <vector>
#include <string>
#include <cmath>
#include <limits>

#include "gmmvb.hpp"
#include "matrixView.hpp"
#include "utilities.hpp"
#include "index.hpp"

using namespace BayesicSpace;
using std::vector;
using std::string;
using std::numeric_limits;

const double GmmVB::lnMaxDbl_ = log( numeric_limits<double>::max() );

GmmVB::GmmVB(const vector<double> *yVec, const double &lambda0, const double &tau0, const double alpha0, const size_t &nPop, const size_t &d, vector<double> *vPopMn, vector<double> *vSm, vector<double> *resp, vector<double> *Nm) : yVec_{yVec}, N_{Nm}, lambda0_{lambda0}, nu0_{static_cast<double>(d)}, tau0_{tau0}, alpha0_{alpha0}, d_{static_cast<double>(d)}, nu0p2_{static_cast<double>(d) + 2.0}, nu0p1_{static_cast<double>(d) + 1.0}, dln2_{static_cast<double>(d)*0.6931471806}, maxIt_{200}, stoppingDiff_{1e-4} {
#ifndef PKG_DEBUG_OFF
	if (yVec->size()%d) {
		throw string("ERROR: Y dimensions not compatible with the number of traits supplied in the GmmVB constructor");
	}
#endif
	const size_t n   = yVec->size()/d;
	const size_t dSq = d*d;

	// Set up correct dimensions
	const size_t popDim = nPop*d;
	if (vPopMn->size() != popDim) {
		vPopMn->clear();
		vPopMn->resize(popDim, 0.0);
	}
	const size_t sDim = dSq*nPop;
	if (vSm->size() != sDim) {
		vSm->clear();
		vSm->resize(sDim, 0.0);
	}
	if (Nm->size() != nPop) {
		Nm->clear();
		Nm->resize(nPop, 0.0);
	}
	const size_t rDim = nPop*n;
	if (resp->size() != rDim) {
		resp->clear();
		resp->resize(rDim, 0.0);
	}
	lnDet_.resize(nPop, 0.0);
	sumDiGam_.resize(nPop, 0.0);

	// Set up the matrix views
	Y_ = MatrixViewConst(yVec, 0, n, d);
	R_ = MatrixView(resp, 0, n, nPop);
	M_ = MatrixView(vPopMn, 0, nPop, d);

	vW_.resize(sDim, 0.0);
	size_t sigInd = 0;
	for (size_t m = 0; m < nPop; m++) {
		S_.push_back( MatrixView(vSm, sigInd, d, d) );
		W_.push_back( MatrixView(&vW_, sigInd, d, d) );
		sigInd += dSq;
	}

	// Initialize values with k-means
	Index popInd(nPop);
	const double smallVal = 0.01/static_cast<double>(nPop - 1);
	kMeans_(Y_, nPop, 50, popInd, M_);
	for (size_t m = 0; m < nPop; m++) {
		for (size_t iRow = 0; iRow < n; iRow++) {
			if (popInd.groupID(iRow) == m){
				R_.setElem(iRow, m, 0.99);
			} else {
				R_.setElem(iRow, m, smallVal);
			}
		}
	}
	//sortPops_();
	mStep_();
	for (auto &w : W_){
		w *= 0.1;
	}
	for (auto &ld : lnDet_){
		ld += d_*log(0.1);
	}
}

GmmVB::GmmVB(GmmVB &&in) : yVec_{in.yVec_}, N_{in.N_}, lambda0_{in.lambda0_}, nu0_{in.nu0_}, tau0_{in.tau0_}, alpha0_{in.alpha0_}, d_{in.d_}, nu0p2_{in.nu0p2_}, nu0p1_{in.nu0p1_}, dln2_{in.dln2_}, maxIt_{in.maxIt_}, stoppingDiff_{in.stoppingDiff_} {
	if (&in != this) {
		Y_        = move(in.Y_);
		M_        = move(in.M_);
		S_        = move(in.S_);
		vW_       = move(in.vW_);
		W_        = move(in.W_);
		lnDet_    = move(in.lnDet_);
		sumDiGam_ = move(in.sumDiGam_);
		R_        = move(in.R_);
		N_        = move(in.N_);

		in.yVec_ = nullptr;
		in.N_    = nullptr;
	}
}

void GmmVB::fitModel(vector<double> &lowerBound) {
	lowerBound.clear();
	for (uint16_t it = 0; it < maxIt_; it++) {
		eStep_();
		mStep_();
		const double curLB = getLowerBound_();
		if ( lowerBound.size() ) {
			if ( fabs( ( curLB - lowerBound.back() )/lowerBound.back() ) <= stoppingDiff_ ) {
				lowerBound.push_back(curLB);
				break;
			}
		}
		lowerBound.push_back(curLB);
	}
	// scale the outputs as necessary
	for (size_t m = 0; m < M_.getNrows(); m++) {  // crossproduct into covariance
		if ( (*N_)[m] > numeric_limits<double>::epsilon() ) {
			S_[m] /= (*N_)[m];
		}
	}
	double dm = 2.0;
	for (size_t m = 2; m <= M_.getNrows(); m++) {  // add ln K! as recommended in Bishop, page 484
		lowerBound.back() += log(dm);
		dm                += 1.0;
	}
	// add the constant
	const double n    = static_cast<double>( Y_.getNrows() );
	const double nPop = static_cast<double>( M_.getNrows() );
	double sumLnGam   = 0.0;
	double k          = 1.0;
	for (size_t kk = 0; kk < Y_.getNcols(); kk++) {
		sumLnGam += nuc_.lnGamma( 0.5*(nu0p1_ - k) );
		k        += 1.0;
	}
	lowerBound.back() += nuc_.lnGamma(nPop*alpha0_) - nPop*nuc_.lnGamma(alpha0_) - nuc_.lnGamma(nPop*alpha0_ + n) - nPop*sumLnGam + 0.5*d_*(nPop*(nu0_ + nu0_*log(tau0_) + log(lambda0_) + 2.6931471806) - n*0.144729886);
}

void GmmVB::eStep_(){
	const size_t d    = Y_.getNcols();
	const size_t N    = Y_.getNrows();
	const size_t Npop = M_.getNrows();
	const size_t Ndim = N*d;
	// start with parameters not varying across individuals
	vector<double> startSum;
	vector<double> lamNmRat;
	vector<double> nuNm;
	for (size_t m = 0; m < R_.getNcols(); m++) {
		const double lNm = lambda0_ + (*N_)[m];
		lamNmRat.push_back( (*N_)[m]/lNm );
		nuNm.push_back( 0.5*(nu0p1_ + (*N_)[m]) );
		startSum.push_back( nuc_.digamma(alpha0_ + (*N_)[m]) + 0.5*(sumDiGam_[m] + lnDet_[m] - d_/lNm) );
	}
	// scale the mean matrix
	vector<double> vMsc(Npop*d, 0.0);
	MatrixView Msc(&vMsc, 0, Npop, d);
	for (size_t jCol = 0; jCol < d; jCol++) {
		for (size_t m = 0; m < Npop; m++) {
			Msc.setElem(m, jCol, M_.getElem(m, jCol)*lamNmRat[m]);
		}
	}
	// calculate crossproducts
	vector<double> vLnRho(N*Npop, 0.0);
	MatrixView lnRho(&vLnRho, 0, N, Npop);
	for (size_t m = 0; m < Npop; m++) {
		vector<double> vArsd(*yVec_);
		MatrixView Arsd(&vArsd, 0, N, d);
		for (size_t jCol = 0; jCol < d; jCol++) {
			for (size_t iRow = 0; iRow < N; iRow ++) {
				Arsd.subtractFromElem(iRow, jCol, Msc.getElem(m, jCol));
			}
		}
		vector<double> vArsdSig(Ndim, 0.0);
		MatrixView ArsdSig(&vArsdSig, 0, N, d);
		Arsd.symm('l', 'r', 1.0, W_[m], 0.0, ArsdSig);
		for (size_t jCol = 0; jCol < d; jCol++) {
			for (size_t iRow = 0; iRow < N; iRow++) {
				lnRho.addToElem(iRow, m, Arsd.getElem(iRow, jCol)*ArsdSig.getElem(iRow, jCol));
			}
		}
		for (size_t iRow = 0; iRow < N; iRow++) {
			const double lnRhoLoc = startSum[m] - nuNm[m]*lnRho.getElem(iRow, m);
			lnRho.setElem(iRow, m, lnRhoLoc);
		}
	}
	for (size_t m = 0; m < Npop; m++) {
		for (size_t iRow = 0; iRow < N; iRow++) {
			double invRjm   = 1.0;
			bool noOverflow = true;
			for (size_t l = 0; l < Npop; l++) {
				if (l == m) {
					continue;
				} else {
					double diff = lnRho.getElem(iRow, l) - lnRho.getElem(iRow, m);
					if (diff >= lnMaxDbl_) {                                        // will overflow right away
						R_.setElem(iRow, m, 0.0);
						noOverflow = false;
						break;
					}
					diff = exp(diff);
					if ( (numeric_limits<double>::max() - invRjm) <= diff) {        // will overflow when I add the new value
						R_.setElem(iRow, m, 0.0);
						noOverflow = false;
						break;
					}
					invRjm += diff;
				}
			}
			if (noOverflow) {
				R_.setElem(iRow, m, 1.0/invRjm);
			}
		}
	}
}

void GmmVB::mStep_() {
	const size_t Npop = M_.getNrows();
	const size_t d    = M_.getNcols();
	const size_t N    = Y_.getNrows();
	R_.colSums(*N_);
	for (size_t m = 0; m < Npop; m++) {
		// calculate aBar_m (weighted mean)
		// updating the mean is more numerically stable than a straight calculation
		for (size_t jCol = 0; jCol < d; jCol++) {
			double aWtMn  = 0.0;
			double weight = 0.0;
			for (size_t iRow = 0; iRow < N; iRow++) {
				nuc_.updateWeightedMean(Y_.getElem(iRow, jCol), R_.getElem(iRow, m), aWtMn, weight);
			}
			M_.setElem(m, jCol, aWtMn);
		}
		// Calculate regular and weighted residuals
		vector<double> vArsd(*yVec_);
		MatrixView Arsd(&vArsd, 0, N, d);
		vector<double> vWtArsd(N*d, 0.0);
		MatrixView wtArsd(&vWtArsd, 0, N, d);
		for (size_t jCol = 0; jCol < d; jCol++) {
			for (size_t iRow = 0; iRow < N; iRow++) {
				Arsd.subtractFromElem( iRow, jCol, M_.getElem(m, jCol) );
				wtArsd.setElem( iRow, jCol, Arsd.getElem(iRow, jCol)*R_.getElem(iRow, m) );
			}
		}
		// unscaled Sm
		Arsd.gemm(true, 1.0, wtArsd, false, 0.0, S_[m]);
		const double lNmRatio = (lambda0_*(*N_)[m])/(lambda0_ + (*N_)[m]);
		// lower triangle of Sigma_m
		for (size_t jD = 0; jD < d; jD++) {
			for (size_t iD = jD; iD < d; iD++) {
				W_[m].setElem( iD, jD, S_[m].getElem(iD, jD) + lNmRatio*M_.getElem(m, iD)*M_.getElem(m, jD) );
			}
		}
		// complete symmetric matrix
		for (size_t jD = 0; jD < d; jD++) {
			for (size_t iD = 0; iD < jD; iD++) {
				W_[m].setElem( iD, jD, W_[m].getElem(jD, iD) );
			}
		}
		// add S_0 (it is diagonal)
		for (size_t kk = 0; kk < d; kk++) {
			W_[m].addToElem(kk, kk, tau0_);
		}
		// invert, trying the faster Cholesky inversion first
		try {
			W_[m].chol();
			W_[m].cholInv();
		} catch (string problem) {
			W_[m].pseudoInv();
		}
	}
}

double GmmVB::getLowerBound_(){
	const size_t Npop = M_.getNrows();
	const size_t d    = M_.getNcols();
	const size_t N    = Y_.getNrows();
	double lwrBound   = 0.0;
	for (size_t m = 0; m < M_.getNrows(); m++) {
		sumDiGam_[m]    = 0.0;
		double sumLnGam = 0.0;
		const double nu0p2Nm = nu0p2_ + (*N_)[m];
		double k = 1.0;
		for (size_t kk = 0; kk < d; kk++) {
			const double arg = (nu0p2Nm - k)/2.0;
			sumDiGam_[m] += nuc_.digamma(arg);
			sumLnGam     += nuc_.lnGamma(arg);
			k += 1.0;
		}
		// add the log-pseudo-determinant
		vector<double> lam;
		vector<double> vU(d*d, 0.0);
		MatrixView U(&vU, 0, d, d);
		W_[m].eigenSafe('l', U, lam);
		lnDet_[m] = 0.0;
		for (auto &l : lam){
			if (l > 0.0) {
				lnDet_[m] += log(l);
			}
		}
		const double psiElmt = 0.5*(sumDiGam_[m] + dln2_ + (nu0p2Nm + 1.0)*lnDet_[m]);
		// add the matrix trace
		double matTr = 0.0;
		vector<double> vSS(d*d, 0.0);
		MatrixView SS(&vSS, 0, d, d);
		W_[m].symm('l', 'l', 1.0, S_[m], 0.0, SS);
		for (size_t kk = 0; kk < d; kk++) {
			matTr += SS.getElem(kk, kk) + tau0_*W_[m].getElem(kk, kk);
		}
		// a_m crossproduct
		vector<double> aS(d, 0.0);
		for (size_t jD = 0; jD < d; jD++) {
			for (size_t iD = 0; iD < d; iD++) {
				aS[jD] += M_.getElem(m, iD)*W_[m].getElem(iD,jD);
			}
		}
		double aSaT = 0.0;
		for (size_t kk = 0; kk < d; kk++) {
			aSaT += aS[kk]*M_.getElem(m, kk);
		}
		const double nu0p1Nm = nu0p1_ + (*N_)[m];
		const double lmNmSm  = lambda0_ + (*N_)[m];
		// the big first sum (in curly brackets in the model document), multiplied by N_m
		const double bigSum  = psiElmt + nu0p1Nm*( (*N_)[m]*lambda0_*(0.5*lambda0_ - (*N_)[m])*aSaT/(lmNmSm*lmNmSm) - 0.5*matTr );
		// r_jm ln(r_jm) sum
		double rLnr = 0.0;
		for (size_t i = 0; i < N; i++) {
			const double r = R_.getElem(i, m);
			if ( r > numeric_limits<double>::epsilon() ) {  // lim x-> 0 (x ln(x) ) = 0
				rLnr += r*log(r);
			}
		}
		// put it all together (+= because we are summing across populations)
		lwrBound += bigSum - rLnr + nuc_.lnGamma(alpha0_ + (*N_)[m]) - 0.5*d_*log(lmNmSm) + sumLnGam;
	}
	return lwrBound;
}

double GmmVB::rowDistance_(const MatrixViewConst &m1, const size_t &row1, const MatrixView &m2, const size_t &row2){
#ifndef PKG_DEBUG_OFF
	if ( m1.getNcols() != m2.getNcols() ) {
		throw string("ERROR: m1 and m2 matrices must have the same number of columns in GmmVB::rowDist_()");
	}
	if ( row1+1 > m1.getNrows() ) {
		throw string("ERROR: row1  index out of bounds in GmmVB::rowDist_()");
	}
	if ( row2+1 > m2.getNrows() ) {
		throw string("ERROR: row2  index out of bounds in GmmVB::rowDist_()");
	}
#endif
	double dist = 0.0;
	for (size_t jCol = 0; jCol < m1.getNcols(); jCol++) {
		double diff = m1.getElem(row1, jCol) - m2.getElem(row2, jCol);
		dist += diff*diff;
	}
	return sqrt(dist);
}

void GmmVB::kMeans_(const MatrixViewConst &X, const size_t &Kclust, const uint32_t &maxIt, Index &x2m, MatrixView &M){
#ifndef PKG_DEBUG_OFF
	if (M.getNrows() != Kclust) {
		throw string("ERROR: Matrix of means must have one row per cluster in GmmVB::kMeans_()");
	}
	if ( X.getNcols() != M.getNcols() ) {
		throw string("ERROR: Matrix of observations must have the same number of columns as the matrix of means in GmmVB::kMeans_()");
	}
	if ( M.getNrows() != x2m.groupNumber() ) {
		throw string("ERROR: observation to cluster index must be the same number of groups as the number of populations in GmmVB::kMeans_()");
	}
#endif
	// initialize M with a random pick of X rows (the MacQueen 1967 method)
	size_t curXind = 0;
	size_t curMind = 0;
	double N       = static_cast<double>(X.getNrows() - 1);   // # of remaining rows
	double n       = static_cast<double>(Kclust);             // # of clusters to be picked
	while( curMind < M.getNrows() ){
		curXind += rng_.vitter(n, N);
		for (size_t jCol = 0; jCol < X.getNcols(); jCol++) {
			M.setElem( curMind, jCol, X.getElem(curXind, jCol) );
		}
		n -= 1.0;
		N -= static_cast<double>(curXind);
		curMind++;
	}
	// Iterate the k-means algorithm
	vector<size_t> sPrevious;             // previous cluster assignment vector
	vector<size_t> sNew(X.getNrows(), 0); // new cluster assignment vector
	for (uint32_t i = 0; i < maxIt; i++) {
		// save the previous S vector
		sPrevious = sNew;
		// assign cluster IDs according to minimal distance
		for (size_t iRow = 0; iRow < X.getNrows(); iRow++) {
			sNew[iRow]  = 0;
			double dist = rowDistance_(X, iRow, M, 0);
			for (size_t iCl = 1; iCl < M.getNrows(); iCl++) {
				double curDist = rowDistance_(X, iRow, M, iCl);
				if (dist > curDist) {
					sNew[iRow] = iCl;
					dist       = curDist;
				}
			}
		}
		x2m.update(sNew);
		// recalculate cluster means
		X.colMeans(x2m, M);
		// calculate the magnitude of cluster assignment change
		double nDiff = 0.0;
		for (size_t i = 0; i < sNew.size(); i++) {
			if (sNew[i] != sPrevious[i] ) {
				nDiff++;
			}
		}
		if ( ( nDiff/static_cast<double>( X.getNrows() ) ) <= 0.1 ) { // fewer than 10% of assignments changed
			break;
		}
	}
}

void GmmVB::sortPops_(){//TODO: fix the sorts; don't seem to be working right at all
	vector<double> firstIdx;                                       // vector with indices of the first high-p elements per population
	for (size_t m = 0; m < R_.getNcols(); m++) {
		for (size_t iRow = 0; iRow < R_.getNrows(); iRow++) {
			if (R_.getElem(iRow, m) >= 0.95){
				firstIdx.push_back(static_cast<double>(iRow));
				break;
			}
		}
	}
	if ( firstIdx.size() < R_.getNcols() ){                      // some populations may have no high-probability individuals
		while ( firstIdx.size() != R_.getNcols() ){
			firstIdx.push_back( R_.getNrows() );                 // add one past the last index, to guarantee that these populations will be put last
		}
	}
	vector<size_t> popIdx;
	//TODO: fix insertion sort
	nuc_.sort(firstIdx, popIdx);
	M_.permuteRows(popIdx);
	R_.permuteCols(popIdx);
}
