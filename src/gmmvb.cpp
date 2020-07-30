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

#include <vector>
#include <string>
#include <cmath>

#include "gmmvb.hpp"
#include "matrixView.hpp"
#include "utilities.hpp"
#include "index.hpp"

using namespace BayesicSpace;
using std::vector;
using std::string;


GmmVB::GmmVB(const vector<double> *yVec, const double &lambda0, const double &nu0, const double &tau0, const double alpha0, const size_t &nPop, const size_t &d, vector<double> *vPopMn, vector<double> *vSm, vector<double> *resp, vector<double> *Nm) : yVec_{yVec}, Nm_{Nm}, lambda0_{lambda0}, nu0_{nu0}, tau0_{tau0}, alpha0_{alpha0}, d_{static_cast<double>(d)}, dl0_{static_cast<double>(d)*lambda0}, nu0p2_{nu0 + 2.0}, nu0p1_{nu0 + 1.0}, nu0mdm1_{nu0 - static_cast<double>(d) - 1.0}, maxIt_{100}, stoppingDiff_{1e-3} {
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

	// Set up the matrix views
	Y_ = MatrixViewConst(yVec, 0, n, d);
	R_ = MatrixView(resp, 0, n, nPop);
	M_ = MatrixView(vPopMn, 0, nPop, d);

	vSigM_.resize(sDim, 0.0);
	size_t sigInd = 0;
	for (size_t m = 0; m < nPop; m++) {
		S_.push_back( MatrixView(vSm, sigInd, d, d) );
		SigM_.push_back( MatrixView(&vSigM_, sigInd, d, d) );
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
	sortPops_();
	mStep_();
}

GmmVB::GmmVB(GmmVB &&in) : yVec_{in.yVec_}, Nm_{in.Nm_}, lambda0_{in.lambda0_}, nu0_{in.nu0_}, tau0_{in.tau0_}, alpha0_{in.alpha0_}, d_{in.d_}, dl0_{in.dl0_}, nu0p2_{in.nu0p2_}, nu0p1_{in.nu0p1_}, nu0mdm1_{in.nu0mdm1_}, maxIt_{in.maxIt_}, stoppingDiff_{in.stoppingDiff_} {
	if (&in != this) {
		Y_     = move(in.Y_);
		M_     = move(in.M_);
		S_     = move(in.S_);
		vSigM_ = move(in.vSigM_);
		SigM_  = move(in.SigM_);
		R_     = move(in.R_);
		Nm_    = move(in.Nm_);

		in.yVec_ = nullptr;
		in.Nm_   = nullptr;
	}
}

double GmmVB::mStep_() {
	const size_t Npop = M_.getNrows();
	const size_t d    = M_.getNcols();
	const size_t N    = Y_.getNrows();
	double lwrBound   = 0.0;
	R_.colSums(*Nm_);
	for (size_t m = 0; m < Npop; m++) {
		// calculate aBar_m (weighted mean)
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
		Arsd.gemm(true, 1.0, wtArsd, false, 1.0, S_[m]);
		const double lNmRatio = (lambda0_*(*Nm_)[m])/(lambda0_ + (*Nm_)[m]);
		// lower triangle of Sigma_m
		for (size_t jD = 0; jD < d; jD++) {
			for (size_t iD = jD; iD < d; iD++) {
				SigM_[m].setElem( iD, jD, S_[m].getElem(iD, jD) + lNmRatio*M_.getElem(m, iD)*M_.getElem(m, jD) );
			}
		}
		// complete symmetric matrix
		for (size_t jD = 0; jD < d; jD++) {
			for (size_t iD = 0; iD < jD; iD++) {
				SigM_[m].setElem( iD, jD, SigM_[m].getElem(jD, iD) );
			}
		}
		// add S_0 (it is diagonal)
		for (size_t kk = 0; kk < d; kk++) {
			SigM_[m].addToElem(kk, kk, tau0_);
		}
		// invert
		SigM_[m].chol();
		SigM_[m].cholInv();
		// complete S_m
		S_[m] /= (*Nm_)[m];
		// calculate the lower bound portion for this population
		double psiElmt = 0.0;
		double k = 1.0;
		for (size_t kk = 0; kk < d; kk++) {
			psiElmt += nuc_.digamma( (nu0p2_ + (*Nm_)[m] - k)/2.0 );
			k += 1.0;
		}
		// add the log-pseudo-determinant
		vector<double> lam;
		vector<double> vU(d*d, 0.0);
		MatrixView U(&vU, 0, d, d);
		SigM_[m].eigenSafe('l', U, lam);
		for (auto &l : lam){
			if (l > 0.0) {
				psiElmt += log(l);
			}
		}
		psiElmt *= (nu0mdm1_ + (*Nm_)[m])/2.0;
		// add the matrix trace
		double matTr = 0.0;
		vector<double> vSS(d*d, 0.0);
		MatrixView SS(&vSS, 0, d, d);
		SigM_[m].symm('l', 'l', 1.0, S_[m], 0.0, SS);
		for (size_t kk = 0; kk < d; kk++) {
			matTr += SS.getElem(kk, kk) + tau0_*SigM_[m].getElem(kk, kk);
		}
		// a_m crossproduct
		vector<double> aS(d, 0.0);
		for (size_t jD = 0; jD < d; jD++) {
			for (size_t iD = 0; iD < d; iD++) {
				aS[jD] += M_.getElem(m, iD)*SigM_[m].getElem(iD,jD);
			}
		}
		double aSaT = 0.0;
		for (size_t kk = 0; kk < d; kk++) {
			aSaT += aS[kk]*M_.getElem(m, kk);
		}
		double lmNmSm = lambda0_ + (*Nm_)[m];
		aSaT *= ( lambda0_*(lambda0_ + (*Nm_)[m]*(*Nm_)[m]) )/(2.0*lmNmSm*lmNmSm);
		// r_jm ln(r_jm) sum
		double rLnr = 0.0;
		for (size_t i = 0; i < N; i++) {
			const double r = R_.getElem(i, m);
			rLnr += r*log(r);
		}
		// put it all together (+= because we are summing across populations
		lwrBound += psiElmt - 0.5*(nu0p1_ + (*Nm_)[m])*(matTr + aSaT) - 0.5*dl0_/lmNmSm - rLnr + nuc_.lnGamma(alpha0_ + (*Nm_)[m]) - 0.5*d_*log(lmNmSm);

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
	double N       = static_cast<double>( X.getNrows() ); // # of remaining rows
	double n       = static_cast<double>(Kclust);         // # of clusters to be picked
	while( curMind < M.getNrows() ){
		curXind += rng_.vitter(n, N);
		for (size_t jCol = 0; jCol < X.getNcols(); jCol++) {
			M.setElem( curMind, jCol, X.getElem(curXind, jCol) );
		}
		n = n - 1.0;
		N = N - static_cast<double>(curXind) + 1.0;
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

void GmmVB::sortPops_(){
	vector<size_t> firstIdx;                                       // vector with indices of the first high-p elements per population
	for (size_t m = 0; m < R_.getNcols(); m++) {
		for (size_t iRow = 0; iRow < R_.getNrows(); iRow++) {
			if (R_.getElem(iRow, m) >= 0.95){
				firstIdx.push_back(iRow);
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
	nuc_.insertionSort(firstIdx, popIdx);
	M_.permuteRows(popIdx);
	R_.permuteCols(popIdx);
}
