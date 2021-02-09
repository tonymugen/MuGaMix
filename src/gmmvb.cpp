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

#include <cstddef>
#include <vector>
#include <string>
#include <cmath>
#include <limits>

#include "gmmvb.hpp"
#include "matrixView.hpp"
#include "bayesicUtilities/utilities.hpp"
#include "bayesicUtilities/index.hpp"

using namespace BayesicSpace;
using std::vector;
using std::string;
using std::numeric_limits;
using std::isnan;

const double GmmVB::lnMaxDbl_ = log( numeric_limits<double>::max() );

// GmmVB methods
GmmVB::GmmVB(const vector<double> *yVec, const double &lambda0, const double &sigmaSq0, const double alpha0, const size_t &nPop, const size_t &d, vector<double> *vPopMn, vector<double> *vSm, vector<double> *resp, vector<double> *Nm) : yVec_{yVec}, N_{Nm}, lambda0_{lambda0}, nu0_{static_cast<double>(d)}, sigmaSq0_{sigmaSq0}, alpha0_{alpha0}, d_{static_cast<double>(d)}, nu0p2_{static_cast<double>(d) + 2.0}, nu0p1_{static_cast<double>(d) + 1.0}, dln2_{static_cast<double>(d) * 0.6931471806}, maxIt_{200}, stoppingDiff_{1e-4} {
#ifndef PKG_DEBUG_OFF
	if (yVec->size()%d) {
		throw string("ERROR: Y dimensions not compatible with the number of traits supplied in the GmmVB constructor");
	}
#endif
	const size_t n   = yVec->size() / d;
	const size_t dSq = d * d;

	// Set up correct dimensions
	const size_t popDim = nPop * d;
	if (vPopMn->size() != popDim) {
		vPopMn->clear();
		vPopMn->resize(popDim, 0.0);
	}
	const size_t sDim = dSq * nPop;
	if (vSm->size() != sDim) {
		vSm->clear();
		vSm->resize(sDim, 0.0);
	}
	if (Nm->size() != nPop) {
		Nm->clear();
		Nm->resize(nPop, 0.0);
	}
	const size_t rDim = nPop * n;
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

	size_t sigInd = 0;
	for (size_t m = 0; m < nPop; m++) {
		W_.push_back( MatrixView(vSm, sigInd, d, d) );
		sigInd += dSq;
	}
}

GmmVB::GmmVB(GmmVB &&in) : yVec_{in.yVec_}, N_{in.N_}, lambda0_{in.lambda0_}, nu0_{in.nu0_}, sigmaSq0_{in.sigmaSq0_}, alpha0_{in.alpha0_}, d_{in.d_}, nu0p2_{in.nu0p2_}, nu0p1_{in.nu0p1_}, dln2_{in.dln2_}, maxIt_{in.maxIt_}, stoppingDiff_{in.stoppingDiff_} {
	if (&in != this) {
		Y_        = move(in.Y_);
		M_        = move(in.M_);
		W_        = move(in.W_);
		lnDet_    = move(in.lnDet_);
		sumDiGam_ = move(in.sumDiGam_);
		R_        = move(in.R_);
		N_        = move(in.N_);

		in.yVec_ = nullptr;
		in.N_    = nullptr;
	}
}

void GmmVB::fitModel(vector<double> &logPost, double &dic) {
	// Initialize values with k-means
	Index popInd( M_.getNrows() );
	const double smallVal = 0.01 / static_cast<double>(M_.getNrows() - 1);
	kMeans_(Y_, M_.getNrows(), 50, popInd, M_);
	for (size_t m = 0; m < M_.getNrows(); m++) {
		for (size_t iRow = 0; iRow < Y_.getNrows(); iRow++) {
			if (popInd.groupID(iRow) == m){
				R_.setElem(iRow, m, 0.99);
			} else {
				R_.setElem(iRow, m, smallVal);
			}
		}
	}
	mStep_();
	// Fit model
	logPost.clear();
	dic = 0.0;
	for (size_t it = 0; it < maxIt_; it++) {
		eStep_();
		mStep_();
		const double curLP = logPost_();
		if ( logPost.size() && (fabs( ( curLP - logPost.back() ) / logPost.back() ) <= stoppingDiff_) ) {
			logPost.push_back(curLP);
			break;
		}
		logPost.push_back(curLP);
	}
	// complete the DIC
	const double nmAlphaN = static_cast<double>( M_.getNrows() ) * alpha0_ + static_cast<double>( Y_.getNrows() );
	double pD = 0.0;
	for (size_t m = 0; m < M_.getNrows(); m++) {
		const double alpha_m = alpha0_ + (*N_)[m];
		pD += (*N_)[m] * (2.0 * ( log(alpha_m) - nuc_.digamma(alpha_m) ) + d_ * log(nu0p1_ + (*N_)[m]) - sumDiGam_[m] + 1.0 / (lambda0_ + (*N_)[m]));
	}
	dic = 2.0 * ( pD - logPost.back() ) - static_cast<double>( Y_.getNrows() ) * ( 2.0 * log(nmAlphaN) - d_ * 0.4515827053 - 4.0 * nuc_.digamma(nmAlphaN) ); // 0.4515827053 is log(pi/2)

	// scale the inverse covariance and invert
	for (size_t m = 0; m < M_.getNrows(); m++) {  // scale the inverse-covariance
		W_[m] *= nu0p1_ + (*N_)[m];
		try {
			W_[m].chol();
			W_[m].cholInv();
		} catch (string problem) {
			W_[m].pseudoInv();
		}
	}
}

void GmmVB::eStep_(){
	const size_t d    = Y_.getNcols();
	const size_t N    = Y_.getNrows();
	const size_t Npop = M_.getNrows();
	const size_t Ndim = N * d;
	// start with parameters not varying across individuals
	vector<double> startSum;
	vector<double> nuNm;
	for (size_t m = 0; m < Npop; m++) {
		const double lNm = lambda0_ + (*N_)[m];
		nuNm.push_back( 0.5 * (nu0p1_ + (*N_)[m]) );
		startSum.push_back( nuc_.digamma(alpha0_ + (*N_)[m]) + 0.5 * (sumDiGam_[m] + lnDet_[m] - d_ / lNm) );
	}
	// calculate crossproducts
	vector<double> vLnRho(N * Npop, 0.0);
	MatrixView lnRho(&vLnRho, 0, N, Npop);
	for (size_t m = 0; m < Npop; m++) {
		vector<double> vArsd(*yVec_);
		MatrixView Arsd(&vArsd, 0, N, d);
		for (size_t jCol = 0; jCol < d; jCol++) {
			for (size_t iRow = 0; iRow < N; iRow ++) {
				Arsd.subtractFromElem(iRow, jCol, M_.getElem(m, jCol));
			}
		}
		vector<double> vArsdSig(Ndim, 0.0);
		MatrixView ArsdSig(&vArsdSig, 0, N, d);
		Arsd.symm('l', 'r', 1.0, W_[m], 0.0, ArsdSig);
		for (size_t jCol = 0; jCol < d; jCol++) {
			for (size_t iRow = 0; iRow < N; iRow++) {
				lnRho.addToElem(iRow, m, Arsd.getElem(iRow, jCol) * ArsdSig.getElem(iRow, jCol));
			}
		}
		for (size_t iRow = 0; iRow < N; iRow++) {
			const double lnRhoLoc = startSum[m] - nuNm[m] * lnRho.getElem(iRow, m);
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
				R_.setElem(iRow, m, 1.0 / invRjm);
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
		// Calculate weighted data
		vector<double> vYsc(*yVec_);
		MatrixView Ysc(&vYsc, 0, N, d);
		for (size_t jCol = 0; jCol < d; jCol++) {
			for (size_t iRow = 0; iRow < N; iRow++) {
				Ysc.multiplyElem( iRow, jCol, R_.getElem(iRow, m) );
			}
		}
		const double lamNm = lambda0_ + (*N_)[m];
		vector<double> mRow;
		Ysc.colSums(mRow);
		for (size_t jCol = 0; jCol < d; jCol++) {
			M_.setElem(m, jCol, mRow[jCol] / lamNm);
		}
		Y_.gemm(true, 1.0, Ysc, false, 0.0, W_[m]);
		// lower triangle of W_m
		for (size_t jD = 0; jD < d; jD++) {
			for (size_t iD = jD; iD < d; iD++) {
				W_[m].subtractFromElem( iD, jD, lamNm * M_.getElem(m, iD) * M_.getElem(m, jD) );
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
			W_[m].addToElem(kk, kk, sigmaSq0_);
		}
		// invert and the log-determinant in one shot
		try {
			W_[m].chol();
			lnDet_[m] = 0.0;
			for (size_t kk = 0; kk < d; kk++) {
				lnDet_[m] += log( W_[m].getElem(kk, kk) );
			}
			lnDet_[m] *= -2.0; // because we did not invert yet
			W_[m].cholInv();
		} catch (string problem) {
			W_[m].pseudoInv(lnDet_[m]);
		}
		// calculate the digamma sum
		const double nu0p2Nm = nu0p2_ + (*N_)[m];
		sumDiGam_[m]         = 0.0;
		double k             = 1.0;
		for (size_t kk = 0; kk < d; kk++) {
			sumDiGam_[m] += nuc_.digamma( 0.5 * (nu0p2Nm - k) );
			k            += 1.0;
		}
	}
}

double GmmVB::logPost_(){
	const size_t d    = Y_.getNcols();
	const size_t N    = Y_.getNrows();
	const size_t Npop = M_.getNrows();
	const size_t Ndim = N * d;
	// start with parameters not varying across individuals
	vector<double> nuNm;
	for (size_t m = 0; m < Npop; m++) {
		const double lNm = lambda0_ + (*N_)[m];
		nuNm.push_back(nu0p1_ + (*N_)[m]);
	}
	// calculate the K matrix (kappa_jm)
	vector<double> vK(N * Npop, 0.0);
	MatrixView K(&vK, 0, N, Npop);
	for (size_t m = 0; m < Npop; m++) {
		vector<double> vArsd(*yVec_);
		MatrixView Arsd(&vArsd, 0, N, d);
		for (size_t jCol = 0; jCol < d; jCol++) {
			for (size_t iRow = 0; iRow < N; iRow ++) {
				Arsd.subtractFromElem(iRow, jCol, M_.getElem(m, jCol));
			}
		}
		vector<double> vArsdSig(Ndim, 0.0);
		MatrixView ArsdSig(&vArsdSig, 0, N, d);
		Arsd.symm('l', 'r', 1.0, W_[m], 0.0, ArsdSig);
		for (size_t jCol = 0; jCol < d; jCol++) {
			for (size_t iRow = 0; iRow < N; iRow++) {
				K.addToElem(iRow, m, Arsd.getElem(iRow, jCol) * ArsdSig.getElem(iRow, jCol));
			}
		}
	}
	// add the scalar values
	vector<double> scSum;
	for (size_t m = 0; m < Npop; m++) {
		scSum.push_back( log(alpha0_ + (*N_)[m]) + 0.5 * (d_ * log(nuNm[m]) + lnDet_[m]) );
	}
	vector<size_t> maxInd(N, 0);                                          // index of the largest kernel value
	vector<double> curMaxVal( N, -numeric_limits<double>::infinity() );   // store the current largest kernel value here
	for (size_t m = 0; m < Npop; m++) {
		for (size_t iRow = 0; iRow < N; iRow++) {
			const double newVal = scSum[m] - 0.5 * nuNm[m] * K.getElem(iRow, m);
			if (newVal > curMaxVal[iRow]) {
				maxInd[iRow]    = m;
				curMaxVal[iRow] = newVal;
			}
			K.setElem(iRow, m, newVal);
		}
	}
	// Subtract the largest column from the rest and sum the exponents row-wise
	vector<double> expRowSums(N, 0.0);
	for (size_t m = 0; m < Npop; m++) {
		for (size_t iRow = 0; iRow < N; iRow++) {
			if (m != maxInd[iRow]) {
				expRowSums[iRow] += exp(K.getElem(iRow, m) - K.getElem(iRow, maxInd[iRow]));
			}
		}
	}
	// sum everything
	double lnP = 0.0;
	for (size_t i = 0; i < N; i++) {
		lnP += curMaxVal[i] + log1p(expRowSums[i]);
	}
	return(lnP);
}

double GmmVB::rowDistance_(const MatrixViewConst &m1, const size_t &row1, const MatrixView &m2, const size_t &row2){
#ifndef PKG_DEBUG_OFF
	if ( m1.getNcols() != m2.getNcols() ) {
		throw string("ERROR: m1 and m2 matrices must have the same number of columns in GmmVB::rowDist_()");
	}
	if ( row1 + 1 > m1.getNrows() ) {
		throw string("ERROR: row1  index out of bounds in GmmVB::rowDist_()");
	}
	if ( row2 + 1 > m2.getNrows() ) {
		throw string("ERROR: row2  index out of bounds in GmmVB::rowDist_()");
	}
#endif
	double dist = 0.0;
	for (size_t jCol = 0; jCol < m1.getNcols(); jCol++) {
		double diff = m1.getElem(row1, jCol) - m2.getElem(row2, jCol);
		dist += diff * diff;
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
			if (sNew[i] != sPrevious[i]) {
				nDiff++;
			}
		}
		if ( ( nDiff / static_cast<double>( X.getNrows() ) ) <= 0.1 ) { // fewer than 10% of assignments changed
			break;
		}
	}
}

// GmmVBmiss methods
GmmVBmiss::GmmVBmiss(vector<double> *yVec, const double &lambda0, const double &sigmaSq0, const double alpha0, const size_t &nPop, const size_t &d, vector<double> *vPopMn, vector<double> *vSm, vector<double> *resp, vector<double> *Nm) : GmmVB(yVec, lambda0, sigmaSq0, alpha0, nPop, d, vPopMn, vSm, resp, Nm) {
	missInd_.resize( Y_.getNcols() );
	MatrixView Ytmp( yVec, 0, Y_.getNrows(), Y_.getNcols() );  // this is to modify yVec
	for (size_t jCol = 0; jCol < Y_.getNcols(); jCol++) {
		for (size_t iRow = 0; iRow < Y_.getNrows(); iRow++) {
			if ( isnan( Y_.getElem(iRow, jCol) ) ) {
				missInd_[jCol].push_back(iRow);
				Ytmp.setElem(iRow, jCol, 0.0);
			}
		}
	}
}

void GmmVBmiss::fitModel(vector<double> &logPost, double &dic){
	// Initialize values with k-means
	Index popInd( M_.getNrows() );
	const double smallVal = 0.01 / static_cast<double>(M_.getNrows() - 1);
	kMeans_(Y_, M_.getNrows(), 50, popInd, M_);
	for (size_t m = 0; m < M_.getNrows(); m++) {
		for (size_t iRow = 0; iRow < Y_.getNrows(); iRow++) {
			if (popInd.groupID(iRow) == m){
				R_.setElem(iRow, m, 0.99);
			} else {
				R_.setElem(iRow, m, smallVal);
			}
		}
	}
	mStep_();
	// Fit model
	logPost.clear();
	dic = 0.0;
	for (size_t it = 0; it < maxIt_; it++) {
		eStep_();
		mStep_();
		const double curLP = logPost_();
		if ( logPost.size() && (fabs( ( curLP - logPost.back() ) / logPost.back() ) <= stoppingDiff_) ) {
			logPost.push_back(curLP);
			break;
		}
		logPost.push_back(curLP);
	}
	// complete the DIC
	const double nmAlphaN = static_cast<double>( M_.getNrows() ) * alpha0_ + static_cast<double>( Y_.getNrows() );
	double pD = 0.0;
	for (size_t m = 0; m < M_.getNrows(); m++) {
		const double alpha_m = alpha0_ + (*N_)[m];
		pD += (*N_)[m] * (2.0 * ( log(alpha_m) - nuc_.digamma(alpha_m) ) + d_ * log(nu0p1_ + (*N_)[m]) - sumDiGam_[m] + 1.0 / (lambda0_ + (*N_)[m]));
	}
	dic = 2.0 * ( pD - logPost.back() ) - static_cast<double>( Y_.getNrows() ) * ( 2.0 * log(nmAlphaN) - d_ * 0.4515827053 - 4.0 * nuc_.digamma(nmAlphaN) ); // 0.4515827053 is log(pi/2)

	// scale the inverse covariance and invert
	for (size_t m = 0; m < M_.getNrows(); m++) {  // scale the inverse-covariance
		W_[m] *= nu0p1_ + (*N_)[m];
		try {
			W_[m].chol();
			W_[m].cholInv();
		} catch (string problem) {
			W_[m].pseudoInv();
		}
	}
}

void GmmVBmiss::eStep_(){
	const size_t d    = Y_.getNcols();
	const size_t N    = Y_.getNrows();
	const size_t Npop = M_.getNrows();
	const size_t Ndim = N * d;
	// start with parameters not varying across individuals
	vector<double> startSum;
	vector<double> nuNm;
	for (size_t m = 0; m < Npop; m++) {
		const double lNm = lambda0_ + (*N_)[m];
		nuNm.push_back( 0.5 * (nu0p1_ + (*N_)[m]) );
		startSum.push_back( nuc_.digamma(alpha0_ + (*N_)[m]) + 0.5 * (sumDiGam_[m] + lnDet_[m] - d_ / lNm) );
	}
	// calculate crossproducts
	vector<double> vLnRho(N * Npop, 0.0);
	MatrixView lnRho(&vLnRho, 0, N, Npop);
	for (size_t m = 0; m < Npop; m++) {
		vector<double> vArsd(*yVec_);
		MatrixView Arsd(&vArsd, 0, N, d);
		for (size_t jCol = 0; jCol < d; jCol++) {
			if ( missInd_[jCol].empty() ) {
				for (size_t iRow = 0; iRow < N; iRow ++) {
					Arsd.subtractFromElem(iRow, jCol, M_.getElem(m, jCol));
				}
			} else {
				size_t curMind = 0;
				for (size_t iRow = 0; iRow < N; iRow ++) {
					if ( ( curMind < missInd_[jCol].size() ) && (missInd_[jCol][curMind] == iRow) ) {
						curMind++;
					} else {
						Arsd.subtractFromElem(iRow, jCol, M_.getElem(m, jCol));
					}
				}
			}
		}
		vector<double> vArsdSig(Ndim, 0.0);
		MatrixView ArsdSig(&vArsdSig, 0, N, d);
		Arsd.symm('l', 'r', 1.0, W_[m], 0.0, ArsdSig);
		for (size_t jCol = 0; jCol < d; jCol++) {
			if ( missInd_[jCol].empty() ) {
				for (size_t iRow = 0; iRow < N; iRow++) {
					lnRho.addToElem(iRow, m, Arsd.getElem(iRow, jCol) * ArsdSig.getElem(iRow, jCol));
				}
			} else {
				size_t curMind = 0;
				for (size_t iRow = 0; iRow < N; iRow++) {
					if ( ( curMind < missInd_[jCol].size() ) && (missInd_[jCol][curMind] == iRow) ) {
						curMind++;
					} else {
						lnRho.addToElem(iRow, m, Arsd.getElem(iRow, jCol) * ArsdSig.getElem(iRow, jCol));
					}
				}
			}
		}
		for (size_t iRow = 0; iRow < N; iRow++) {
			const double lnRhoLoc = startSum[m] - nuNm[m] * lnRho.getElem(iRow, m);
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
				R_.setElem(iRow, m, 1.0 / invRjm);
			}
		}
	}
}

void GmmVBmiss::mStep_(){
	const size_t Npop = M_.getNrows();
	const size_t d    = M_.getNcols();
	const size_t N    = Y_.getNrows();
	R_.colSums(*N_);
	for (size_t m = 0; m < Npop; m++) {
		// Calculate weighted data
		vector<double> vYsc(*yVec_);
		MatrixView Ysc(&vYsc, 0, N, d);
		for (size_t jCol = 0; jCol < d; jCol++) {
			for (size_t iRow = 0; iRow < N; iRow++) {
				Ysc.multiplyElem( iRow, jCol, R_.getElem(iRow, m) );
			}
		}
		vector<double> lamNm(d, lambda0_);
		for (size_t jCol = 0; jCol < d; jCol++) {
			if ( missInd_[jCol].empty() ) {
				lamNm[jCol] += (*N_)[m];
			} else {
				size_t curMind = 0;
				for (size_t iRow = 0; iRow < N; iRow++) {
					if ( ( curMind < missInd_[jCol].size() ) && (missInd_[jCol][curMind] == iRow) ) {
						curMind++;
					} else {
						lamNm[jCol] += R_.getElem(iRow, m);
					}
				}
			}
		}
		vector<double> mRow;
		Ysc.colSums(missInd_, mRow);
		for (size_t jCol = 0; jCol < d; jCol++) {
			M_.setElem(m, jCol, mRow[jCol] / lamNm[jCol]);
		}
		Y_.gemm(true, 1.0, Ysc, false, 0.0, W_[m]);
		// lower triangle of W_m
		for (size_t jD = 0; jD < d; jD++) {
			for (size_t iD = jD; iD < d; iD++) {
				W_[m].subtractFromElem( iD, jD, sqrt(lamNm[iD]) * sqrt(lamNm[jD]) * M_.getElem(m, iD) * M_.getElem(m, jD) );
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
			W_[m].addToElem(kk, kk, sigmaSq0_);
		}
		// invert and the log-determinant in one shot
		try {
			W_[m].chol();
			lnDet_[m] = 0.0;
			for (size_t kk = 0; kk < d; kk++) {
				lnDet_[m] += log( W_[m].getElem(kk, kk) );
			}
			lnDet_[m] *= -2.0; // because we did not invert yet
			W_[m].cholInv();
		} catch (string problem) {
			W_[m].pseudoInv(lnDet_[m]);
		}
		// calculate the digamma sum
		const double nu0p2Nm = nu0p2_ + (*N_)[m];
		sumDiGam_[m]         = 0.0;
		double k             = 1.0;
		for (size_t kk = 0; kk < d; kk++) {
			sumDiGam_[m] += nuc_.digamma( 0.5 * (nu0p2Nm - k) );
			k            += 1.0;
		}
	}
}

double GmmVBmiss::logPost_(){
	const size_t d    = Y_.getNcols();
	const size_t N    = Y_.getNrows();
	const size_t Npop = M_.getNrows();
	const size_t Ndim = N * d;
	// start with parameters not varying across individuals
	vector<double> nuNm;
	for (size_t m = 0; m < Npop; m++) {
		const double lNm = lambda0_ + (*N_)[m];
		nuNm.push_back(nu0p1_ + (*N_)[m]);
	}
	// calculate the K matrix (kappa_jm)
	vector<double> vK(N * Npop, 0.0);
	MatrixView K(&vK, 0, N, Npop);
	for (size_t m = 0; m < Npop; m++) {
		vector<double> vArsd(*yVec_);
		MatrixView Arsd(&vArsd, 0, N, d);
		for (size_t jCol = 0; jCol < d; jCol++) {
			if ( missInd_[jCol].empty() ) {
				for (size_t iRow = 0; iRow < N; iRow ++) {
					Arsd.subtractFromElem(iRow, jCol, M_.getElem(m, jCol));
				}
			} else {
				size_t curMind = 0;
				for (size_t iRow = 0; iRow < N; iRow ++) {
					if ( ( curMind < missInd_[jCol].size() ) && (missInd_[jCol][curMind] == iRow) ) {
						curMind++;
					} else {
						Arsd.subtractFromElem(iRow, jCol, M_.getElem(m, jCol));
					}
				}
			}
		}
		vector<double> vArsdSig(Ndim, 0.0);
		MatrixView ArsdSig(&vArsdSig, 0, N, d);
		Arsd.symm('l', 'r', 1.0, W_[m], 0.0, ArsdSig);
		for (size_t jCol = 0; jCol < d; jCol++) {
			if ( missInd_[jCol].empty() ) {
				for (size_t iRow = 0; iRow < N; iRow++) {
					K.addToElem(iRow, m, Arsd.getElem(iRow, jCol) * ArsdSig.getElem(iRow, jCol));
				}
			} else {
				size_t curMind = 0;
				for (size_t iRow = 0; iRow < N; iRow++) {
					if ( ( curMind < missInd_[jCol].size() ) && (missInd_[jCol][curMind] == iRow) ) {
						curMind++;
					} else {
						K.addToElem(iRow, m, Arsd.getElem(iRow, jCol) * ArsdSig.getElem(iRow, jCol));
					}
				}
			}
		}
	}
	// add the scalar values
	vector<double> scSum;
	for (size_t m = 0; m < Npop; m++) {
		scSum.push_back( log(alpha0_ + (*N_)[m]) + 0.5 * (d_ * log(nuNm[m]) + lnDet_[m]) );
	}
	vector<size_t> maxInd(N, 0);                                          // index of the largest kernel value
	vector<double> curMaxVal( N, -numeric_limits<double>::infinity() );   // store the current largest kernel value here
	for (size_t m = 0; m < Npop; m++) {
		for (size_t iRow = 0; iRow < N; iRow++) {
			const double newVal = scSum[m] - 0.5 * nuNm[m] * K.getElem(iRow, m);
			if (newVal > curMaxVal[iRow]) {
				maxInd[iRow]    = m;
				curMaxVal[iRow] = newVal;
			}
			K.setElem(iRow, m, newVal);
		}
	}
	// Subtract the largest column from the rest and sum the exponents row-wise
	vector<double> expRowSums(N, 0.0);
	for (size_t m = 0; m < Npop; m++) {
		for (size_t iRow = 0; iRow < N; iRow++) {
			if (m != maxInd[iRow]) {
				expRowSums[iRow] += exp(K.getElem(iRow, m) - K.getElem(iRow, maxInd[iRow]));
			}
		}
	}
	// sum everything
	double lnP = 0.0;
	for (size_t i = 0; i < N; i++) {
		lnP += curMaxVal[i] + log1p(expRowSums[i]);
	}
	return(lnP);
}

double GmmVBmiss::rowDistance_(const MatrixViewConst &m1, const size_t &row1, const MatrixView &m2, const size_t &row2, const vector<size_t> &presInd){
#ifndef PKG_DEBUG_OFF
	if ( m1.getNcols() != m2.getNcols() ) {
		throw string("ERROR: m1 and m2 matrices must have the same number of columns in GmmVBmiss::rowDist_()");
	}
	if ( row1 + 1 > m1.getNrows() ) {
		throw string("ERROR: row1  index out of bounds in GmmVBmiss::rowDist_()");
	}
	if ( row2 + 1 > m2.getNrows() ) {
		throw string("ERROR: row2  index out of bounds in GmmVBmiss::rowDist_()");
	}
#endif
	double dist = 0.0;
	for (auto &pv : presInd){
		double diff = m1.getElem(row1, pv) - m2.getElem(row2, pv);
		dist       += diff * diff;
	}
	return sqrt(dist);
}

void GmmVBmiss::kMeans_(const MatrixViewConst &X, const size_t &Kclust, const uint32_t &maxIt, Index &x2m, MatrixView &M){
#ifndef PKG_DEBUG_OFF
	if (M.getNrows() != Kclust) {
		throw string("ERROR: Matrix of means must have one row per cluster in GmmVBmiss::kMeans_()");
	}
	if ( X.getNcols() != M.getNcols() ) {
		throw string("ERROR: Matrix of observations must have the same number of columns as the matrix of means in GmmVBmiss::kMeans_()");
	}
	if ( M.getNrows() != x2m.groupNumber() ) {
		throw string("ERROR: observation to cluster index must be the same number of groups as the number of populations in GmmVBmiss::kMeans_()");
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
		vector<size_t> missRowInd(Y_.getNcols(), 0);
		for (size_t iRow = 0; iRow < X.getNrows(); iRow++) {
			sNew[iRow]  = 0;
			vector<size_t> presInd;
			for (size_t j = 0; j < Y_.getNcols(); j++) {
				if ( ( missInd_[j].size() ) && ( missRowInd[j] < missInd_[j].size() ) && (missInd_[j][ missRowInd[j] ] == iRow) ) {
					missRowInd[j]++;
				} else {
					presInd.push_back(j);
				}
			}
			double dist = rowDistance_(X, iRow, M, 0, presInd);
			for (size_t iCl = 1; iCl < M.getNrows(); iCl++) {
				double curDist = rowDistance_(X, iRow, M, iCl, presInd);
				if (dist > curDist) {
					sNew[iRow] = iCl;
					dist       = curDist;
				}
			}
		}
		x2m.update(sNew);
		// recalculate cluster means
		X.colMeans(x2m, missInd_, M);
		// calculate the magnitude of cluster assignment change
		double nDiff = 0.0;
		for (size_t i = 0; i < sNew.size(); i++) {
			if (sNew[i] != sPrevious[i]) {
				nDiff++;
			}
		}
		if ( ( nDiff / static_cast<double>( X.getNrows() ) ) <= 0.1 ) { // fewer than 10% of assignments changed
			break;
		}
	}
}

