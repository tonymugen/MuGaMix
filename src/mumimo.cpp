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

/// Gaussian mixture models
/** \file
 * \author Anthony J. Greenberg
 * \copyright Copyright (c) 2019 Anthony J. Greenberg
 * \version 1.0
 *
 * Class implementation to generate Markov chains for inference of Gaussian mixture models. Dual-averaging NUTS and Metropolis samplers for parameters groups are included within a Gibbs sampler.
 *
 */

#include <cstddef>
#include <cstring>
#include <vector>
#include <string>
#include <cmath>
#include <limits> // for numeric_limits

#include "mumimo.hpp"
#include "gmmvb.hpp"
#include "bayesicUtilities/index.hpp"
#include "bayesicUtilities/random.hpp"
#include "bayesicMatrix/matrixView.hpp"
#include "bayesicSamplers/danuts.hpp"
#include "bayesicSamplers/metropolis.hpp"

using std::vector;
using std::string;
using std::to_string;
using std::numeric_limits;
using std::isnan;
using std::move;
using namespace BayesicSpace;

// MumiNR methods

const double MumiNR::lnMaxDbl_ = log( numeric_limits<double>::max() );

MumiNR::MumiNR(const vector<double> *yVec, const vector<double> *lnpVec, const size_t &d, const size_t &Ngrp, const double &tau0, const double &nu0, const double &invAsq) : Model(), yVec_{yVec}, tau0_{tau0}, nu0_{nu0}, invAsq_{invAsq} {
#ifndef PKG_DEBUG_OFF
	if (yVec->size()%d) {
		throw string("ERROR: Y dimensions not compatible with the number of traits supplied in the MumiNR constructor");
	}
#endif
	const size_t n = yVec->size() / d;
#ifndef PKG_DEBUG_OFF
	if (lnpVec->size() != n * Ngrp){
		throw string("ERROR: ln(P) dimensions not compatible with the number of groups supplied in the MumiNR constructor");
	}
#endif
	Y_   = MatrixViewConst(yVec, 0, n, d);
	lnP_ = MatrixViewConst(lnpVec, 0, n, Ngrp);

	size_t dSq = d * d;
	size_t dd  = d * (d - 1) / 2;
	vLa_.resize(dSq * Ngrp, 0.0);
	La_.resize(Ngrp);
	size_t startLa = 0;
	size_t curLaInd = (Ngrp+1) * d;
	for (auto &la : La_){
		la = MatrixView(&vLa_, startLa, d, d);
		startLa += dSq;
		for (size_t k = 0; k < d; k++) {
			la.setElem(k, k, 1.0);
		}
		LaInd_.push_back(curLaInd);
		curLaInd += dd;
	}
	size_t curTaInd = LaInd_.back() + dd;
	for (size_t m = 0; m < Ngrp; m++) {
		TaInd_.push_back(curTaInd);
		curTaInd += d;
	}
	TgInd_ = TaInd_.back() + d;
	NAnd_  = nu0_ + 2.0 * static_cast<double>(d);
	NGnd_  = static_cast<double>(Ngrp) + nu0_ + 2.0 * static_cast<double>(d);
	nxnd_  = nu0_ * NAnd_;
}

MumiNR::MumiNR(MumiNR &&in) {
	if (this != &in) {
		yVec_   = in.yVec_;
		Y_      = move(in.Y_);
		lnP_    = move(in.lnP_);
		tau0_   = in.tau0_;
		nu0_    = in.nu0_;
		invAsq_ = in.invAsq_;
		La_     = move(in.La_);
		vLa_    = move(in.vLa_);
		LaInd_  = move(in.LaInd_);
		TaInd_  = move(in.TaInd_);
		TgInd_  = in.TgInd_;
		NAnd_   = in.NAnd_;
		NGnd_   = in.NGnd_;

		in.yVec_ = nullptr;
	}
}

MumiNR& MumiNR::operator=(MumiNR &&in) {
	if (this != &in) {
		yVec_   = in.yVec_;
		Y_      = move(in.Y_);
		lnP_    = move(in.lnP_);
		tau0_   = in.tau0_;
		nu0_    = in.nu0_;
		invAsq_ = in.invAsq_;
		La_     = move(in.La_);
		vLa_    = move(in.vLa_);
		LaInd_  = move(in.LaInd_);
		TaInd_  = move(in.TaInd_);
		TgInd_  = in.TgInd_;
		NAnd_   = in.NAnd_;
		NGnd_   = in.NGnd_;

		in.yVec_ = nullptr;
	}
	return *this;
}

void MumiNR::expandISvec_(const vector<double> &theta) const{
	size_t aInd = LaInd_[0];                                               // index of the Lx lower triangle in the input vector
	for (auto &la : La_){
		for (size_t jCol = 0; jCol < Y_.getNcols() - 1; jCol++) {          // the last column is all 0, except the last element = 1.0
			for (size_t iRow = jCol + 1; iRow < Y_.getNcols(); iRow++) {
				la.setElem(iRow, jCol, theta[aInd]);
				aInd++;
			}
		}
	}
}

double MumiNR::logPost(const vector<double> &theta) const{
	// make L matrices
	expandISvec_(theta);
	const size_t N    = Y_.getNrows();
	const size_t d    = Y_.getNcols();
	const size_t Ngrp = lnP_.getNcols();
	MatrixViewConst Mp(&theta, 0, Ngrp, d);
	MatrixViewConst mu(&theta, Ngrp * d, 1, d);                                // overall mean

	// calculate T_A
	vector< vector<double> > Ta(Ngrp);
	vector<double> SAlDet(Ngrp, 0.0);                                          // Sig_A,m log-determinant
	for (size_t m = 0; m < Ngrp; m++) {
		for (size_t k = TaInd_[m]; k < TaInd_[m] + d; k++) {
			Ta[m].push_back( exp(theta[k]) );
			SAlDet[m] += theta[k];
		}
	}
	// calculate T_G
	vector<double> Tg;
	for (size_t k = TgInd_; k < theta.size(); k++) {                           // the T_G component is at the very end
		Tg.push_back( exp(theta[k]) );
	}
	// set up the matrix of group kernels
	vector<double> vKm(N * Ngrp, 0.0);
	MatrixView Km( &vKm, 0, N, Ngrp );
	for (size_t m = 0; m < Ngrp; m++) {                                        // m is the group index as in the model description document
		vector<double> vResid(*yVec_);                                         // copy over Y_
		MatrixView mResid = MatrixView(&vResid, 0, N, d);                      // mResid now has the Y values
		for (size_t jCol = 0; jCol < d; jCol++) {
			for (size_t iRow = 0; iRow < N; iRow++) {
				mResid.subtractFromElem( iRow, jCol, Mp.getElem(m, jCol) );    // mResid now Y - mu_m
			}
		}
		mResid.trm('l', 'r', false, true, 1.0, La_[m]);                        // mResid now (Y-mu_m)L_A
		for (size_t jCol = 0; jCol < d; jCol++) {
			for (size_t iRow = 0; iRow < N; iRow++) {
				double rsd = mResid.getElem(iRow, jCol);
				Km.addToElem(iRow, m, Ta[m][jCol] * rsd * rsd);               // (Y-mu_m)L_A T_A L_A^T(Y - mu_g)^T
			}
		}
	}

	for (size_t iRow = 0; iRow < N; iRow++) {                                  // ln(p) + 0.5*(lambda_m - Km) + for the first group
		const double diff = lnP_.getElem(iRow, 0) + 0.5 * ( SAlDet[0] - Km.getElem(iRow, 0) );
		Km.setElem(iRow, 0, diff);
	}
	for (size_t m = 1; m < Ngrp; m++) {                                        // Km[,2...] now the difference with the first group
		for (size_t iRow = 0; iRow < N; iRow++) {
			const double diff = lnP_.getElem(iRow, m) + 0.5 * ( SAlDet[m] - Km.getElem(iRow, m) ) - Km.getElem(iRow, 0);
			Km.setElem(iRow, m, diff);
		}
	}
	double addMMsum = 0.0;                                                     // sum of the additive kernel will go here
	for (size_t iRow = 0; iRow < N; iRow++) {                                  // sacrificing the tight loop to make numerical safety happen
		double regSum = 0.0;
		double bigSum = 0.0;                                                   // will be used if large values of Km are encountered
		for (size_t m = 1; m < Ngrp; m++) {
			const double df = Km.getElem(iRow, m);
			if (df >= 100) {                                                   // well into approximation territory, but don't want to do this too often
				if (bigSum > 0.0) {                                            // something already added
					double ldif = bigSum - df;
					if ( (ldif > 0.0) && (ldif <= 5.0) ) {                     // over 5.0 the correction is unnecessary regardless of the df or bigSum value
						bigSum += log1p( exp(-ldif) );
					} else if ( (ldif < 0.0) && (ldif >= -5.0) ) {
						bigSum = df + log1p( exp(ldif) );
					} else if (ldif < 0.0) {
						bigSum = df;
					} // or leave bigSum as is
				} else {
					bigSum = df;
				}
			} else if (bigSum > 0.0) {
				if (df >= 95) {
					bigSum += log1p( exp(df - bigSum) );
				}
				// otherwise do nothing
			} else {
				if (regSum <= 1e260) { // do not bother adding any more if regSum is too large to prevent overflow; 1e250 ~ exp(600)
					regSum += exp(df);
				}
			}
		}
		if (bigSum > 0.0) {
			addMMsum += Km.getElem(iRow, 0) + bigSum;
		} else {
			addMMsum += Km.getElem(iRow, 0) + log1p(regSum);
		}
	}
	// M[p] crossproduct trace
	double mTrace = 0.0;
	for (size_t jCol = 0; jCol < d; ++jCol) {
		double dp = 0.0;
		for (size_t iRow = 0; iRow < Ngrp; ++iRow) {
			double diff = Mp.getElem(iRow, jCol) - mu.getElem(0, jCol);
			dp += diff * diff;
		}
		mTrace += Tg[jCol] * dp;
		mTrace += dp;
	}
	double pTrace = 0.0;
	for (size_t jCol = 0; jCol < d; jCol++) {
		pTrace += mu.getElem(0, jCol) * mu.getElem(0, jCol);
	}
	pTrace *= tau0_;
	// Sum of log-determinants
	double ldetSumA = 0.0;
	for (auto &det : SAlDet){
		ldetSumA += det;
	}
	ldetSumA *= NAnd_;
	double ldetSumP = 0.0;
	for (size_t k = 0; k < d; k++) {
		ldetSumP += theta[TgInd_ + k];
	}
	ldetSumP *= NGnd_;
	// Calculate the prior inverse-covariance components; k and m are as in the derivation document; doing the L_E and L_A in one pass
	// first element has just the diagonal
	double isPrior = 0.0;
	for (size_t m = 0; m < Ngrp; m++) {
		isPrior += log(nu0_ * Ta[m][0] + invAsq_);
		for (size_t k = 1; k < d; k++) {                                       // k starts from the second element (k=1)
			double sA = 0.0;
			for (size_t l = 0; l <= k - 1; l++) {                              // the <= is intentional; excluding only l == k
				sA += Ta[m][l] * La_[m].getElem(k, l) * La_[m].getElem(k, l);
			}
			sA += Ta[m][k];
			isPrior += log(nu0_ * sA + invAsq_);
		}
	}
	for (size_t k = 0; k < d; k++) {
		isPrior += log(nu0_ * Tg[k] + invAsq_);
	}
	isPrior *= NAnd_;
	// now sum to get the log-posterior
	return addMMsum - 0.5 * (mTrace + pTrace - ldetSumA - ldetSumP + isPrior);
}
void MumiNR::gradient(const vector<double> &theta, vector<double> &grad) const {
	expandISvec_(theta);
	if ( grad.size() ) {
		grad.clear();
	}
	grad.resize(theta.size(), 0.0);
	const size_t N      = Y_.getNrows();
	const size_t d      = Y_.getNcols();
	const size_t trDim  = d * (d+1) / 2;
	const size_t dSq    = d * d;
	const size_t Ngrp   = lnP_.getNcols();
	const size_t Ydim   = Y_.getNrows() * Y_.getNcols();
	const size_t Mdim   = lnP_.getNcols() * Y_.getNcols();
	const size_t GrpDim = N * Ngrp;
	MatrixViewConst M(&theta, 0, Ngrp, d);
	MatrixViewConst mu(&theta, Mdim, 1, d);

	// Matrix views of the gradient
	MatrixView gM(&grad, 0, Ngrp, d);
	MatrixView gmu(&grad, Mdim, 1, d); // overall mean

	// L_AxT_A
	vector<double> vLATA(dSq * Ngrp, 0.0);
	vector<MatrixView> LATA;
	for (size_t m = 0; m < Ngrp; m++) {
		LATA.push_back( MatrixView(&vLATA, dSq * m, d, d) );
	}
	// calculate T_A
	vector< vector<double> > Ta(Ngrp);
	vector<double> SAlDet(Ngrp, 0.0);                               // Sig_A,m log-determinant
	for (size_t m = 0; m < Ngrp; m++) {
		for (size_t k = TaInd_[m]; k < TaInd_[m] + d; k++) {
			Ta[m].push_back( exp(theta[k]) );
			SAlDet[m] += theta[k];
		}
	}
	// calculate T_G
	vector<double> Tg;
	for (size_t k = TgInd_; k < theta.size(); k++) {                // the T_G component is at the very end
		Tg.push_back( exp(theta[k]) );
	}
	for (size_t m = 0; m < Ngrp; m++) {
		for (size_t jCol = 0; jCol < d - 1; jCol++) {
			LATA[m].setElem(jCol, jCol, Ta[m][jCol]);
			for (size_t iRow = jCol + 1; iRow < d; iRow++) {
				LATA[m].setElem(iRow, jCol, La_[m].getElem(iRow, jCol) * Ta[m][jCol]);
			}
		}
		LATA[m].setElem( d - 1, d - 1, Ta[m].back() );
	}

	// set up the matrix of group kernels
	vector<double> vKappa(GrpDim, 0.0);
	MatrixView Kappa(&vKappa, 0, N, Ngrp);
	vector<double> vResid(Ydim * Ngrp, 0.0);                                             // will have Y residuals (needed later)
	vector<MatrixView> mResid(Ngrp);
	vector<double> vResidLaTa(Ydim * Ngrp, 0.0);                                         // will have Y residuals, multiplied by L_A,m T_A,m with each group mean
	vector<MatrixView> mResidLaTa(Ngrp);
	for (size_t m = 0; m < Ngrp; m++) {                                                  // m is the group index as in the model description document
		memcpy( vResid.data() + Ydim * m, yVec_->data(), Ydim * sizeof(double) );
		mResid[m] = MatrixView(&vResid, Ydim * m, N, d);
		for (size_t jCol = 0; jCol < d; jCol++) {
			for (size_t iRow = 0; iRow < N; iRow++) {
				mResid[m].subtractFromElem( iRow, jCol, M.getElem(m, jCol) );            // mResid now Y - mu_m
			}
		}
		memcpy( vResidLaTa.data() + Ydim * m, vResid.data() + Ydim * m, Ydim * sizeof(double) );
		mResidLaTa[m] = MatrixView(&vResidLaTa, Ydim * m, N, d);
		mResidLaTa[m].trm('l', 'r', false, true, 1.0, La_[m]);                           // mResidLaTa now (Y-mu_m)L_A,m
		for (size_t jCol = 0; jCol < d; jCol++) {
			for (size_t iRow = 0; iRow < N; iRow++) {
				const double rsd   = mResidLaTa[m].getElem(iRow, jCol);
				const double tArsd = Ta[m][jCol] * rsd;
				mResidLaTa[m].setElem(iRow, jCol, tArsd);                                // mResidLaTa now (Y-mu_m)L_A,m T_A,m
				Kappa.addToElem(iRow, m,  tArsd * rsd);                                  // (Y-mu_m)L_A,m T_A,m L_A,m^T(Y - mu_g)^T
			}
		}
	}
	for (size_t m = 0; m < Ngrp; m++) {                                                  // Km now ln(p) + 0.5*(lamdba_m - Km)
		for (size_t iRow = 0; iRow < N; iRow++) {
			const double diff = lnP_.getElem(iRow, m) + 0.5 * ( SAlDet[m] - Kappa.getElem(iRow, m) );
			Kappa.setElem(iRow, m, diff);
		}
	}
	vector<double> kappaRow(Ngrp, 0.0);
	for (size_t iRow = 0; iRow < N; iRow++) {                                            // tight loop not efficient here
		for (size_t m = 0; m < Ngrp; m++) {
			kappaRow[m] = Kappa.getElem(iRow, m);                                        // save the untransformed Kappa elements here
		}
		for (size_t m = 0; m < Ngrp; m++){
			double denominator = 1.0;
			for (size_t p = 0; p < Ngrp; p++) {
				if (p == m) {
					continue;
				}
				const double localExponent = kappaRow[p] - kappaRow[m];
				if (localExponent >= lnMaxDbl_) {                                        // exponentiation will overflow, the inverse is 0
					denominator = nan("");
					break;
				}
				const double valueToAdd = exp(localExponent);
				if ( (numeric_limits<double>::max() - denominator) <= valueToAdd ) {      // adding the current value will overflow
					denominator = nan("");
					break;
				}
				denominator += valueToAdd;
			}
			if ( isnan(denominator) ) {
				Kappa.setElem(iRow, m, 0.0);
			} else {
				Kappa.setElem( iRow, m, 1.0 / denominator );                              // 1 + already included at initialization; Kappa now the exponent ratio
			}
		}
	}
	for (size_t m = 0; m < Ngrp; m++) {                                                   // scale (1+exp(kernDiff))^{-1}*(a_j - mu_m)LaTa
		for (size_t jCol = 0; jCol < d; jCol++) {
			for (size_t iRow = 0; iRow < N; iRow++) {
				mResidLaTa[m].multiplyElem( iRow, jCol, Kappa.getElem(iRow, m) );
			}
		}
	}
	// iSig partial derivatives; do these first because the M gradient requires mResidLaTa to be multiplied by La^T
	vector<double> kappaColSums;
	Kappa.colSums(kappaColSums);
	for (size_t m = 0; m < Ngrp; m++) {
		// start with the L_A,m prior
		vector<double> vechLwA;                                                      // vech(L^w_A,m)
		vector<double> weights(d, 0.0);                                              // will become a d-vector of weights (each element corresponding to a row of L_X; the first element is weighted T_A,m[1,1])
		for (size_t jCol = 0; jCol < d - 1; jCol++) {                                // nothing to be done for the last column (it only has a diagonal element)
			for (size_t iRow = jCol + 1; iRow < d; iRow++) {
				vechLwA.push_back( LATA[m].getElem(iRow, jCol) );
				weights[iRow] += vechLwA.back() * La_[m].getElem(iRow, jCol);        // unweighted for now
			}
		}
		for (size_t k = 0; k < d; k++) {
			weights[k] = nu0_ * (weights[k] + Ta[m][k]) + invAsq_;
		}
		size_t vechInd = 0;
		for (size_t jCol = 0; jCol < d - 1; jCol++) {
			for (size_t iRow = jCol + 1; iRow < d; iRow++) {
				vechLwA[vechInd] = vechLwA[vechInd] / weights[iRow];
				vechInd++;
			}
		}
		// weighted sum of the (Y_j-mu_m) dot products
		vector<double> vATDP(dSq, 0.0);                                              // sum of transposed A residual dot products
		MatrixView ATDP(&vATDP, 0, d, d);
		mResidLaTa[m].gemm(true, 1.0, mResid[m], false, 0.0, ATDP);
		vechInd = 0;
		for (size_t jCol = 0; jCol < d - 1; jCol++) {
			for (size_t iRow = jCol + 1; iRow < d; iRow++) {
				grad[LaInd_[m] + vechInd] = -ATDP.getElem(iRow, jCol) - nxnd_ * vechLwA[vechInd];
				vechInd++;
			}
		}
		// The T_A,m gradient
		// L_A,m^T (weighted a_j dot product)* L_A,m T_A,m
		ATDP.trm('l', 'l', true, true, 1.0, La_[m]);
		// now sum everything and store the result in the gradient vector
		const double dDub = 2.0 * static_cast<double>(d);
		for (size_t k = 0; k < d; k++) {
			grad[TaInd_[m] + k] = 0.5 * (nu0_ + dDub + kappaColSums[m] - ATDP.getElem(k, k) - nxnd_ * Ta[m][k] / weights[k]);
		}
	}
	// M partial derivatives
	for (size_t m = 0; m < Ngrp; m++) {
		mResidLaTa[m].trm('l', 'r', true, true, 1.0, La_[m]);                          // mResid now Kappa_m(Y-mu_m)L_A,m T_A,m L_A,m^T
		for (size_t jCol = 0; jCol < d; jCol++) {
			for (size_t iRow = 0; iRow < N; iRow++) {
				gM.addToElem( m, jCol, mResidLaTa[m].getElem(iRow, jCol) );
			}
		}
	}
	// Group mean residual
	vector<double> vGRPresid(theta.begin(), theta.begin()+Mdim);
	MatrixView GRPresid(&vGRPresid, 0, Ngrp, d);
	for (size_t jCol = 0; jCol < d; jCol++) {
		for (size_t iRow = 0; iRow < Ngrp; iRow++) {
			GRPresid.subtractFromElem( iRow, jCol, mu.getElem(0, jCol) );
		}
	}
	for (size_t jCol = 0; jCol < d; jCol++) {
		for (size_t iRow = 0; iRow < Ngrp; iRow++) {
			gM.subtractFromElem(iRow, jCol, GRPresid.getElem(iRow, jCol) * Tg[jCol]);
		}
	}
	// mu partial derivatives
	vector <double> PresSum; // colSums will resize
	GRPresid.colSums(PresSum);
	for (size_t jCol = 0; jCol < d; jCol++) {
		PresSum[jCol] -= mu.getElem(0, jCol) * tau0_;
	}
	for (size_t jCol = 0; jCol < d; jCol++) {
		gmu.setElem(0, jCol, PresSum[jCol]);
	}
	// The T_G gradient
	vector<double> prDotPrd(d, 0.0);
	for (size_t jCol = 0; jCol < d; jCol++) {
		for (size_t iRow = 0; iRow < Ngrp; iRow++) {
			const double rsd = GRPresid.getElem(iRow, jCol);
			prDotPrd[jCol] += rsd * rsd;
		}
	}
	// now sum everything and store the result in the gradient vector
	for (size_t k = 0; k < d; k++) {
		grad[TgInd_ + k] = 0.5 * ( NGnd_ - Tg[k] * ( prDotPrd[k] + nxnd_ / (nu0_ * Tg[k] + invAsq_) ) );
	}
}

// MumiLoc methods
const double MumiLoc::pSumCutOff_ = 0.003;

MumiLoc::MumiLoc(const vector<double> *yVec, const vector<double> *iSigVec, const vector<Index> *hierInd, const double &tau, const size_t &nGrps, const double &tauPrPhi, const double &alphaPr) : Model(), hierInd_{hierInd}, tau0_{tau}, iSigTheta_{iSigVec}, Ngrp_{nGrps}, tauPrPhi_{tauPrPhi}, alphaPr_{alphaPr} {
	const size_t n = (*hierInd_)[0].size();
#ifndef PKG_DEBUG_OFF
	if (yVec->size()%n) {
		throw string("ERROR: Y dimensions not compatible with the number of data points implied by the replicate factor");
	}
#endif
	const size_t d  = yVec->size() / n;
	Y_              = MatrixViewConst(yVec, 0, n, d);

	vLx_.resize(2 * d * d, 0.0);
	Le_ = MatrixView(&vLx_, 0, d, d);
	La_ = MatrixView(&vLx_, d * d, d, d);
	for (size_t k = 0; k < d; k++) {
		Le_.setElem(k, k, 1.0);
		La_.setElem(k, k, 1.0);
	}
	size_t trLen = d * (d-1) / 2;
	fTeInd_      = trLen;
	fLaInd_      = trLen + d;
	fTaInd_      = fLaInd_ + trLen;
	fTgInd_      = fTaInd_ + d;
	PhiBegInd_   = ( (*hierInd_)[0].groupNumber() + Ngrp_ + 1 ) * d;
}

MumiLoc::MumiLoc(MumiLoc &&in) {
	if (this != &in) {
		Y_         = move(in.Y_);
		tau0_      = in.tau0_;
		hierInd_   = in.hierInd_;
		Le_        = move(in.Le_);
		La_        = move(in.La_);
		vLx_       = move(in.vLx_);
		fTeInd_    = in.fTeInd_;
		fLaInd_    = in.fLaInd_;
		fTaInd_    = in.fTaInd_;
		PhiBegInd_ = in.PhiBegInd_;
		Ngrp_      = in.Ngrp_;
		tauPrPhi_ = in.tauPrPhi_;

		in.hierInd_   = nullptr;
		in.iSigTheta_ = nullptr;
	}
}

MumiLoc& MumiLoc::operator=(MumiLoc &&in) {
	if (this != &in) {
		Y_         = move(in.Y_);
		tau0_      = in.tau0_;
		hierInd_   = in.hierInd_;
		Le_        = move(in.Le_);
		La_        = move(in.La_);
		vLx_       = move(in.vLx_);
		fTeInd_    = in.fTeInd_;
		fLaInd_    = in.fLaInd_;
		fTaInd_    = in.fTaInd_;
		PhiBegInd_ = in.PhiBegInd_;
		Ngrp_      = in.Ngrp_;
		tauPrPhi_ = in.tauPrPhi_;

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
	const size_t Ydim = Y_.getNrows() * Y_.getNcols();
	MatrixViewConst A( &theta, 0, Nln, Y_.getNcols() );
	MatrixViewConst Mp( &theta, Nln * Y_.getNcols(), Ngrp_, Y_.getNcols() );
	MatrixViewConst mu( &theta, (Nln+Ngrp_) * Y_.getNcols(), 1, Y_.getNcols() ); // overall mean
	MatrixViewConst Phi( &theta, PhiBegInd_, Nln, Ngrp_ );

	// Calculate the residual Y - ZA matrix
	vector<double> vResid(Ydim, 0.0);
	MatrixView mResid( &vResid, 0, Y_.getNrows(), Y_.getNcols() );
	for (size_t jCol = 0; jCol  < Y_.getNcols(); ++jCol) {
		for (size_t iRow = 0; iRow < Y_.getNrows(); ++ iRow) {
			double diff =  Y_.getElem(iRow, jCol) - A.getElem( (*hierInd_)[0].groupID(iRow), jCol );
			mResid.setElem(iRow, jCol, diff); // Y - ZA
		}
	}
	// multiply by L_E
	mResid.trm('l', 'r', false, true, 1.0, Le_);
	// Now calculate the trace of the RL_ET_EL_^TR^T matrix
	double eTrace = 0.0;
	for (size_t jCol = 0; jCol < Y_.getNcols(); jCol++) {
		double dp = 0.0;
		for (size_t iRow = 0; iRow < Y_.getNrows(); iRow++) {
			dp += mResid.getElem(iRow, jCol) * mResid.getElem(iRow, jCol);
		}
		eTrace   += exp( (*iSigTheta_)[fTeInd_ + jCol] ) * dp;
	}

	// backtransform the logit-p_jp, calculate row sums and sum of squares
	vector<double> vP(Phi.getNrows() * Phi.getNcols(), 0.0);
	MatrixView P( &vP, 0, Phi.getNrows(), Phi.getNcols() );
	vector<double> pGrpSum(Nln, 0.0);
	double sumPhiSq = 0.0;
	for (size_t m = 0; m < Phi.getNcols(); m++) {
		for (size_t iRow = 0; iRow < Phi.getNrows(); iRow++) {
			double phi     = Phi.getElem(iRow, m);
			sumPhiSq      += phi * phi;
			double p       = nuc_.logistic(phi);
			pGrpSum[iRow] += p;
			P.setElem(iRow, m, p);
		}
	}
	// Re-weight the P; it may be possible to optimize this further by dividing by weight once after aTrace completion
	for (size_t m = 0; m < Phi.getNcols(); m++) {
		for (size_t iRow = 0; iRow < Phi.getNrows(); iRow++) {
			if (pGrpSum[iRow] <= pSumCutOff_){  // approximation when all p_j are small
				double aSum = 0.0;
				double phiM = Phi.getElem(iRow, m);
				for (size_t jCol = 0; jCol < Phi.getNcols(); jCol++) {
					if (jCol == m){
						aSum += 1.0;
					} else {
						aSum += exp(Phi.getElem(iRow, jCol) - phiM);
					}
					P.setElem(iRow, m, 1.0 / aSum);
				}
			} else {
				P.divideElem(iRow, m, pGrpSum[iRow]);
			}
		}
	}
	vector<double> pLNsum(P.getNcols(), 0.0);
	for (size_t m = 0; m < P.getNcols(); m++) {
		for (size_t iRow = 0; iRow < P.getNrows(); iRow++) {
			pLNsum[m] += P.getElem(iRow, m);
		}
	}
	double gpSum = 0.0;
	for (auto &s : pLNsum){
		gpSum += nuc_.lnGamma(s + alphaPr_);
	}
	// Clear the Y residuals and re-use for A residuals
	vResid.clear();
	vector<double> AtraceVec(Nln, 0.0); // accumulate A trace values here
	// calculate T_A
	vector<double> Ta;
	for (size_t k = fTaInd_; k < fTgInd_; k++) {
		Ta.push_back( exp( (*iSigTheta_)[k] ) );
	}
	for (size_t m = 0; m < Ngrp_; m++) {                                             // m is the group index as in the model description document
		vector<double> locAtr(Nln, 0.0);
		vResid.assign( theta.begin(), theta.begin() + A.getNrows() * A.getNcols() ); // copy over A
		mResid = MatrixView( &vResid, 0, A.getNrows(), A.getNcols() );               // mResid now has the A values
		for (size_t jCol = 0; jCol < A.getNcols(); jCol++) {
			for (size_t iRow = 0; iRow < A.getNrows(); iRow++) {
				mResid.subtractFromElem( iRow, jCol, Mp.getElem(m, jCol) );          // mResid now A - mu_m
			}
		}
		mResid.trm('l', 'r', false, true, 1.0, La_);                                 // mResid now (A-mu_m)L_A
		for (size_t jCol = 0; jCol < A.getNcols(); jCol++) {
			for (size_t iRow = 0; iRow < A.getNrows(); iRow++) {
				double rsd    = mResid.getElem(iRow, jCol);
				locAtr[iRow] += Ta[jCol] * rsd * rsd;                                // (A-mu_m)L_A T_A L_A^T(A - mu_g)^T
			}
		}
		for (size_t j = 0; j < Nln; j++) {
			AtraceVec[j] += P.getElem(j, m) * locAtr[j];                             // P_m(A-mu_m)L_A T_A L_A^T(A-mu_m)^T
		}
	}
	double aTrace = 0.0;
	for (auto &a : AtraceVec){
		aTrace += a;
	}
	// M[p] crossproduct trace
	double mTrace = 0.0;
	for (size_t jCol = 0; jCol < Mp.getNcols(); ++jCol) {
		double dp = 0.0;
		for (size_t iRow = 0; iRow < Mp.getNrows(); ++iRow) {
			double diff = Mp.getElem(iRow, jCol) - mu.getElem(0, jCol);
			dp += diff * diff;
		}
		mTrace += exp( (*iSigTheta_)[fTgInd_ + jCol] ) * dp;
	}
	double pTrace = 0.0;
	for (size_t jCol = 0; jCol < Y_.getNcols(); jCol++) {
		pTrace += mu.getElem(0, jCol) * mu.getElem(0, jCol);
	}
	pTrace *= tau0_;
	// now sum to get the log-posterior
	return -0.5 * (eTrace + aTrace + tauPrPhi_ * sumPhiSq + mTrace + pTrace) + gpSum;
}

void MumiLoc::gradient(const vector<double> &theta, vector<double> &grad) const{
	expandISvec_();
	if ( grad.size() ) {
		grad.clear();
	}
	grad.resize(theta.size(), 0.0);
	const size_t Nln  = (*hierInd_)[0].groupNumber();
	const size_t Ydim = Y_.getNrows() * Y_.getNcols();
	const size_t Adim = Nln * Y_.getNcols();
	const size_t Mdim = Ngrp_ * Y_.getNcols();
	MatrixViewConst A( &theta, 0, Nln, Y_.getNcols() );
	MatrixViewConst M( &theta, Adim, Ngrp_, Y_.getNcols() );
	MatrixViewConst mu( &theta, Adim + Mdim, 1, Y_.getNcols() ); // overall mean
	MatrixViewConst Phi(&theta, PhiBegInd_, Nln, Ngrp_);

	// Matrix views of the gradient
	MatrixView gA( &grad, 0, Nln, Y_.getNcols() );
	MatrixView gM( &grad, Adim, Ngrp_, Y_.getNcols() );
	MatrixView gmu( &grad, Adim + Mdim, 1, Y_.getNcols() ); // overall mean
	MatrixView gPhi(&grad, PhiBegInd_, Nln, Ngrp_);

	// Calculate the residual Y - ZA matrix
	vector<double> vResid(Ydim, 0.0);
	MatrixView mResid( &vResid, 0, Y_.getNrows(), Y_.getNcols() );
	for (size_t jCol = 0; jCol  < Y_.getNcols(); ++jCol) {
		for (size_t iRow = 0; iRow < Y_.getNrows(); ++ iRow) {
			double diff =  Y_.getElem(iRow, jCol) - A.getElem( (*hierInd_)[0].groupID(iRow), jCol);
			mResid.setElem(iRow, jCol, diff); // Y - ZA
		}
	}

	// Make precision matrices
	vector<double> vISigE(Y_.getNcols() * Y_.getNcols(), 0.0);
	MatrixView iSigE( &vISigE, 0, Y_.getNcols(), Y_.getNcols() );
	// L_ExT_E
	vector<double> Tx;
	for (size_t k = fTeInd_; k < fLaInd_; k++) {
		Tx.push_back( exp( (*iSigTheta_)[k] ) );
	}
	for (size_t jCol = 0; jCol < Y_.getNcols() - 1; jCol++) {
		iSigE.setElem(jCol, jCol, Tx[jCol]);
		for (size_t iRow = jCol + 1; iRow < Y_.getNcols(); iRow++) {
			iSigE.setElem(iRow, jCol, Le_.getElem(iRow, jCol) * Tx[jCol]);
		}
	}
	// last element of the vector is the lower right corner element
	vISigE.back() = Tx.back();
	iSigE.trm('l', 'r', true, true, 1.0, Le_);
	vector<double> vISigA(Y_.getNcols() * Y_.getNcols(), 0.0);
	MatrixView iSigA( &vISigA, 0, Y_.getNcols(), Y_.getNcols() );
	for (size_t k = 0; k < iSigA.getNcols(); k++) {
		Tx[k] = exp( (*iSigTheta_)[k + fTaInd_] );
	}
	for (size_t jCol = 0; jCol < Y_.getNcols() - 1; jCol++) {
		iSigA.setElem(jCol, jCol, Tx[jCol]);
		for (size_t iRow = jCol + 1; iRow < Y_.getNcols(); iRow++) {
			iSigA.setElem(iRow, jCol, La_.getElem(iRow, jCol) * Tx[jCol]);
		}
	}
	vISigA.back() = Tx.back();
	iSigA.trm('l', 'r', true, true, 1.0, La_);
	// (Y - ZA)Sig[E]^-1
	vector<double> vResISE(Ydim, 0.0);
	MatrixView mResISE( &vResISE, 0, Y_.getNrows(), Y_.getNcols() );
	mResid.symm('l', 'r', 1.0, iSigE, 0.0, mResISE);

	// backtransform the logit-p_jp and calculate row sums
	vector<double> vP( Phi.getNrows() * Phi.getNcols() );
	MatrixView P( &vP, 0, Phi.getNrows(), Phi.getNcols() );
	vector<double> pGrpSum(Nln, 0.0);
	for (size_t m = 0; m < Phi.getNcols(); m++) {
		for (size_t iRow = 0; iRow < Phi.getNrows(); iRow++) {
			double p       = nuc_.logistic( Phi.getElem(iRow, m) );
			pGrpSum[iRow] += p;
			P.setElem(iRow, m, p);
		}
	}
	// Re-weight p_jm = p_jm/sum_m(p_jm)
	for (size_t m = 0; m < Phi.getNcols(); m++) {
		for (size_t iRow = 0; iRow < Phi.getNrows(); iRow++) {
			if (pGrpSum[iRow] <= pSumCutOff_){  // approximation when all p_j are small
				double aSum = 0.0;
				double phiM = Phi.getElem(iRow, m);
				for (size_t jCol = 0; jCol < Phi.getNcols(); jCol++) {
					if (jCol == m){
						aSum += 1.0;
					} else {
						aSum += exp(Phi.getElem(iRow, jCol) - phiM);
					}
					P.setElem(iRow, m, 1.0 / aSum);
				}
			} else {
				P.divideElem(iRow, m, pGrpSum[iRow]);
			}
		}
	}
	vector<double> digamLNsum;  // will be the digamma of line-wise sums; colSums will expand the vector to the right size
	P.colSums(digamLNsum);
	for (auto &l : digamLNsum){
		l = nuc_.digamma(l + alphaPr_);
	}
	// Z^T(Y - ZA - XB)Sig[E]^-1 store in gA
	mResISE.colSums( (*hierInd_)[0], gA );
	vector<double> kernSum(P.getNrows(), 0.0);
	for (size_t m = 0; m < Ngrp_; m++) {
		vector<double> vAresid(theta.begin(), theta.begin()+Adim);                                      // copying A to the residual matrix
		MatrixView Aresid( &vAresid, 0, A.getNrows(), A.getNcols() );
		for (size_t jCol = 0; jCol < A.getNcols(); jCol++) {
			for (size_t iRow = 0; iRow < A.getNrows(); iRow++) {
				Aresid.subtractFromElem( iRow, jCol, M.getElem(m, jCol) );                              // A - mu_m
			}
		}
		vector<double> vAresISA(A.getNrows() * A.getNcols(), 0.0);
		MatrixView AresISA( &vAresISA, 0, A.getNrows(), A.getNcols() );
		Aresid.symm('l', 'r', 1.0, iSigA, 0.0, AresISA);                                                // (A - mu_m)Sigma^{-1}_A

		for (size_t jCol = 0; jCol < A.getNcols(); jCol++) {
			for (size_t iRow = 0; iRow < A.getNrows(); iRow++) {
				gPhi.addToElem( iRow, m, Aresid.getElem(iRow, jCol) * AresISA.getElem(iRow, jCol) );    // calculate the (j,m) kernel and store in gPhi to calculate phi partials later
			}
		}
		for (size_t iRow = 0; iRow < A.getNrows(); iRow++) {
			kernSum[iRow] += P.getElem(iRow, m) * gPhi.getElem(iRow, m);                                // add the kernels together
		}

		for (size_t jCol = 0; jCol < A.getNcols(); jCol++) {
			for (size_t iRow = 0; iRow < A.getNrows(); iRow++) {
				AresISA.multiplyElem( iRow, jCol, P.getElem(iRow, m) );                                 // P_m(A - mu_m)Sigma^{-1}_A
			}
		}
		// finish A partial derivatives
		for (size_t jCol = 0; jCol < A.getNcols(); jCol++) {
			for (size_t iRow = 0; iRow < A.getNrows(); iRow++) {
				gA.subtractFromElem( iRow, jCol, AresISA.getElem(iRow, jCol) );                         // subtracting each group's P_m(A - mu_m)Sigma^{-1}_A
			}
		}
		// sum the A residuals into gM rows
		for (size_t jCol = 0; jCol < A.getNcols(); jCol++) {
			for (size_t iRow = 0; iRow < A.getNrows(); iRow++) {
				gM.addToElem( m, jCol, AresISA.getElem(iRow, jCol) );
			}
		}
	}
	// kernel_m - sum(kernel_l)
	for (size_t m = 0; m < Ngrp_; m++) {
		for (size_t iRow = 0; iRow < P.getNrows(); iRow++) {
			gPhi.subtractFromElem(iRow, m, kernSum[iRow]);
		}
	}
	// M partial derivatives
	// make tau_p
	vector<double> tauP;
	for (size_t k = fTgInd_; k < iSigTheta_->size(); k++) {
		tauP.push_back( exp( (*iSigTheta_)[k] ) );
	}
	// Group mean residual
	vector<double> vGRPresid(theta.begin()+Adim, theta.begin()+Adim+Mdim);
	MatrixView GRPresid( &vGRPresid, 0, M.getNrows(), M.getNcols() );
	for (size_t jCol = 0; jCol < GRPresid.getNcols(); jCol++) {
		for (size_t iRow = 0; iRow < GRPresid.getNrows(); iRow++) {
			GRPresid.subtractFromElem( iRow, jCol, mu.getElem(0, jCol) );
			GRPresid.multiplyElem(iRow, jCol, tauP[jCol]); // (M-mu)T_M
		}
	}
	// complete the M gradient
	for (size_t jCol = 0; jCol < M.getNcols(); jCol++) {
		for (size_t iRow = 0; iRow < M.getNrows(); iRow++) {
			gM.subtractFromElem( iRow, jCol, GRPresid.getElem(iRow, jCol) );  // gM already has the summed A residuals
		}
	}
	// mu partial derivatives
	vector <double> PresSum; // colSums will resize
	GRPresid.colSums(PresSum);
	for (size_t jCol = 0; jCol < M.getNcols(); jCol++) {
		PresSum[jCol] -= mu.getElem(0, jCol) * tau0_;
	}
	for (size_t jCol = 0; jCol < gmu.getNcols(); jCol++) {
		gmu.setElem(0, jCol, PresSum[jCol]);
	}
	// Phi partial derivatives
	for (size_t m = 0; m < Ngrp_; m++) {
		for (size_t iRow = 0; iRow < gPhi.getNrows(); iRow++) {
			// gPhi already has the weighted kernel product sums (with group sums subtracted) stored from before; t_A element missing because we have one Sigma_A
			double phi     = -0.5 * gPhi.getElem(iRow, m);
			double wtDGsum = 0.0;
			for (size_t mp = 0; mp < Ngrp_; mp++) {
				wtDGsum += P.getElem(iRow, mp) * digamLNsum[mp];
			}
			phi     += digamLNsum[m] - wtDGsum;
			double p = P.getElem(iRow, m);
			phi     *= p * (1.0 - p);
			gPhi.setElem( iRow, m, phi - tauPrPhi_ * Phi.getElem(iRow, m) );
		}
	}
}

// MumiISig methods
MumiISig::MumiISig(const vector<double> *yVec, const vector<double> *vTheta, const vector<Index> *hierInd, const double &nu0, const double &invAsq, const size_t &nGrps) : Model(), hierInd_{hierInd}, nu0_{nu0}, invAsq_{invAsq} {
	const size_t N = (*hierInd_)[0].size(); // first index is data to lines
#ifndef PKG_DEBUG_OFF
	if (yVec->size()%N) {
		throw string("MumiISig constructor ERROR: vectorized data length not divisible by number of data points");
	}
#endif
	const size_t d = yVec->size() / N;
#ifndef PKG_DEBUG_OFF
	if (vTheta->size()%d) {
		throw string("MumiISig constructor ERROR: vectorized parameter set length not divisible by number of traits");
	}
#endif
	Y_ = MatrixViewConst(yVec, 0, N, d);
	const size_t Nln = (*hierInd_)[0].groupNumber();
	A_   = MatrixViewConst(vTheta, 0, Nln, d);
	Mp_  = MatrixViewConst(vTheta, Nln * d, nGrps, d);
	mu_  = MatrixViewConst(vTheta, (Nln+nGrps) * d, 1, d);
	Phi_ = MatrixViewConst(vTheta, (Nln+nGrps+1) * d, Nln, nGrps);
	vLx_.resize(2 * d * d, 0.0);
	Le_ = MatrixView(&vLx_, 0, d, d);
	La_ = MatrixView(&vLx_, d * d, d, d);
	for (size_t k = 0; k < d; k++) {
		Le_.setElem(k, k, 1.0);
		La_.setElem(k, k, 1.0);
	}
	size_t trLen = d * (d-1) / 2;
	fTeInd_      = trLen;
	fLaInd_      = trLen + d;
	fTaInd_      = fLaInd_ + trLen;
	fTgInd_      = fTaInd_ + d;
	nxnd_        = nu0_ * ( nu0_ + 2.0 * static_cast<double>(d) );
	Nnd_         = static_cast<double>( Y_.getNrows() ) + nu0_ + 2.0 * static_cast<double>(d);
	NAnd_        = static_cast<double>( A_.getNrows() ) + nu0_ + 2.0 * static_cast<double>(d);
	NGnd_        = static_cast<double>(nGrps) + nu0_ + 2.0 * static_cast<double>(d);
}

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
		Phi_     = move(in.Phi_);
		vLx_     = move(in.vLx_);
		Le_      = MatrixView( &vLx_, 0, Y_.getNcols(), Y_.getNcols() );
		La_      = MatrixView( &vLx_, Y_.getNcols() * Y_.getNcols(), Y_.getNcols(), Y_.getNcols() );
		fTeInd_  = in.fTeInd_;
		fLaInd_  = in.fLaInd_;
		fTaInd_  = in.fTaInd_;
		nxnd_    = in.nxnd_;
		Nnd_     = in.Nnd_;
		NAnd_    = in.NAnd_;
		NGnd_    = in.NGnd_;

		in.hierInd_ = nullptr;
	}
}

MumiISig& MumiISig::operator=(MumiISig &&in) {
	if (this != &in) {
		hierInd_ = in.hierInd_;
		nu0_     = in.nu0_;
		invAsq_  = in.invAsq_;
		Y_       = move(in.Y_);
		A_       = move(in.A_);
		B_       = move(in.B_);
		Mp_      = move(in.Mp_);
		mu_      = move(in.mu_);
		Phi_     = move(in.Phi_);
		vLx_     = move(in.vLx_);
		Le_      = MatrixView( &vLx_, 0, Y_.getNcols(), Y_.getNcols() );
		La_      = MatrixView( &vLx_, Y_.getNcols() * Y_.getNcols(), Y_.getNcols(), Y_.getNcols() );
		fTeInd_  = in.fTeInd_;
		fLaInd_  = in.fLaInd_;
		fTaInd_  = in.fTaInd_;
		fTgInd_  = in.fTgInd_;
		nxnd_    = in.nxnd_;
		Nnd_     = in.Nnd_;
		NAnd_    = in.NAnd_;
		NGnd_    = in.NGnd_;

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
	vector<double> vResid(Y_.getNcols() * Y_.getNrows(), 0.0);
	MatrixView mResid( &vResid, 0, Y_.getNrows(), Y_.getNcols() );
	for (size_t jCol = 0; jCol  < Y_.getNcols(); ++jCol) {
		for (size_t iRow = 0; iRow < Y_.getNrows(); ++ iRow) {
			double diff = Y_.getElem(iRow, jCol) - A_.getElem( (*hierInd_)[0].groupID(iRow), jCol);
			mResid.setElem(iRow, jCol, diff); // Y - ZA
		}
	}
	// multiply by L_E
	mResid.trm('l', 'r', false, true, 1.0, Le_);
	// Now calculate the trace of the RL_ET_EL_^TR^T matrix
	double eTrace = 0.0;
	for (size_t jCol = 0; jCol < Y_.getNcols(); jCol++) {
		double dp = 0.0;
		for (size_t iRow = 0; iRow < Y_.getNrows(); iRow++) {
			dp += mResid.getElem(iRow, jCol) * mResid.getElem(iRow, jCol);
		}
		eTrace += exp(viSig[fTeInd_ + jCol]) * dp;
	}

	// backtransform the logit-p_jp and sum
	vector<double> vP(Phi_.getNrows() * Phi_.getNcols(), 0.0);
	MatrixView P( &vP, 0, Phi_.getNrows(), Phi_.getNcols() );
	vector<double> pGrpSum(Phi_.getNrows(), 0.0);
	for (size_t m = 0; m < Phi_.getNcols(); m++) {
		for (size_t iRow = 0; iRow < Phi_.getNrows(); iRow++) {
			double p       = nuc_.logistic( Phi_.getElem(iRow, m) );
			pGrpSum[iRow] += p;
			P.setElem(iRow, m, p);
		}
	}
	for (size_t m = 0; m < Phi_.getNcols(); m++) {
		for (size_t iRow = 0; iRow < Phi_.getNrows(); iRow++) {
			P.divideElem(iRow, m, pGrpSum[iRow]);
		}
	}

	// Clear the Y residuals and re-use for A residuals
	vResid.clear();
	vResid.resize(A_.getNrows() * A_.getNcols(), 0.0);
	mResid = MatrixView( &vResid, 0, A_.getNrows(), A_.getNcols() );
	vector<double> AtraceVec(A_.getNrows(), 0.0); // accumulate A trace values here
	// calculate T_A
	vector<double> Ta;
	for (size_t k = fTaInd_; k < fTgInd_; k++) {
		Ta.push_back( exp(viSig[k]) );
	}
	for (size_t m = 0; m < Phi_.getNcols(); m++) {                                 // m is the group index as in the model description document
		vector<double> locAtr(A_.getNrows(), 0.0);
		for (size_t jCol = 0; jCol < A_.getNcols(); jCol++) {
			for (size_t iRow = 0; iRow < A_.getNrows(); iRow++) {
				double diff = A_.getElem(iRow, jCol) - Mp_.getElem(m, jCol);
				mResid.setElem(iRow, jCol, diff);                                  // mResid now A - mu_m
			}
		}
		mResid.trm('l', 'r', false, true, 1.0, La_);                               // mResid now (A-mu_m)L_A
		for (size_t jCol = 0; jCol < A_.getNcols(); jCol++) {
			for (size_t iRow = 0; iRow < A_.getNrows(); iRow++) {
				double rsd    = mResid.getElem(iRow, jCol);
				locAtr[iRow] += Ta[jCol] * rsd * rsd;                                  // (A-mu_m)L_A T_A L_A^T(A - mu_g)^T
			}
		}
		for (size_t j = 0; j < A_.getNrows(); j++) {
			AtraceVec[j] += P.getElem(j, m) * locAtr[j];                             // P_m(A-mu_m)L_A T_A L_A^T(A-mu_m)^T
		}
	}
	double aTrace = 0.0;
	for (auto &a : AtraceVec){
		aTrace += a;
	}
	// M[p] crossproduct trace
	double trM = 0.0;
	for (size_t jCol = 0; jCol < Mp_.getNcols(); ++jCol) {
		double dp = 0.0;
		for (size_t iRow = 0; iRow < Mp_.getNrows(); ++iRow) {
			double diff = Mp_.getElem(iRow, jCol) - mu_.getElem(0, jCol);
			dp += diff * diff;
		}
		trM += exp(viSig[fTgInd_ + jCol]) * dp;
	}
	// Sum of log-determinants
	double ldetSumE = 0.0;
	double ldetSumA = 0.0;
	double ldetSumP = 0.0;
	for (size_t k = 0; k < Y_.getNcols(); k++) {
		ldetSumE += viSig[fTeInd_ + k];
		ldetSumA += viSig[fTaInd_ + k];
		ldetSumP += viSig[fTgInd_ + k];
	}
	ldetSumE *= Nnd_;
	ldetSumA *= NAnd_;
	ldetSumP *= NGnd_;
	// Calculate the prior components; k and m are as in the derivation document; doing the L_E and L_A in one pass
	// first element has just the diagonal
	double pTrace = log(nu0_ * exp(viSig[fTeInd_]) + invAsq_) + log(nu0_ * exp(viSig[fTaInd_]) + invAsq_);
	for (size_t k = 1; k < Le_.getNcols(); k++) { // k starts from the second element (k=1)
		double sE = 0.0;
		double sA = 0.0;
		for (size_t m = 0; m <= k - 1; m++) { // the <= is intentional; excluding only m = k
			sE += exp(viSig[fTeInd_ + m]) * Le_.getElem(k, m) * Le_.getElem(k, m);
			sA += exp(viSig[fTaInd_ + m]) * La_.getElem(k, m) * La_.getElem(k, m);
		}
		sE += exp(viSig[fTeInd_ + k]);
		sA += exp(viSig[fTaInd_ + k]);
		pTrace += log(nu0_ * sE + invAsq_) + log(nu0_ * sA + invAsq_) + log(nu0_ * exp(viSig[fTgInd_ + k]) + invAsq_);
	}
	pTrace *= nu0_ + 2.0 * static_cast<double>( Y_.getNcols() );
	return -0.5 * (eTrace + aTrace + trM - ldetSumE - ldetSumA - ldetSumP + pTrace);
}

void MumiISig::gradient(const vector<double> &viSig, vector<double> &grad) const{
	// expand the element vector to make the L matrices
	expandISvec_(viSig);
	if ( grad.size() ){
		grad.clear();
	}
	grad.resize(viSig.size(), 0.0);
	// Calculate the Y residuals
	vector<double> vResid(Y_.getNcols() * Y_.getNrows(), 0.0);
	MatrixView mResid( &vResid, 0, Y_.getNrows(), Y_.getNcols() );
	for (size_t jCol = 0; jCol  < Y_.getNcols(); ++jCol) {
		for (size_t iRow = 0; iRow < Y_.getNrows(); ++ iRow) {
			double diff =  Y_.getElem(iRow, jCol) - A_.getElem( (*hierInd_)[0].groupID(iRow), jCol );
			mResid.setElem(iRow, jCol, diff); // Y - ZA
		}
	}
	vector<double> vRtR(Y_.getNcols() * Y_.getNcols(), 0.0);
	MatrixView mRtR( &vRtR, 0, Y_.getNcols(), Y_.getNcols() );
	mResid.syrk('l', 1.0, 0.0, mRtR);
	vector<double> vRtRLT(Y_.getNcols() * Y_.getNcols(), 0.0);
	MatrixView mRtRLT( &vRtRLT, 0, Y_.getNcols(), Y_.getNcols() );
	Le_.symm('l', 'l', 1.0, mRtR, 0.0, mRtRLT); // R^TRL_E; R = Y - ZA
	// make a vector of T_X (provided values are on the log scale; will use for T_E and T_A)
	vector<double> Tx;
	for (size_t k = fTeInd_; k < fLaInd_; k++) {
		Tx.push_back( exp(viSig[k]) );
	}
	// mutiply by T_E (the whole matrix because I will need it for left-multiplication later)
	for (size_t jCol = 0; jCol < Y_.getNcols(); jCol++) {
		for (size_t iRow = 0; iRow < Y_.getNcols(); iRow++) {
			double prod = mRtRLT.getElem(iRow, jCol) * Tx[jCol];
			mRtRLT.setElem(iRow, jCol, prod);
		}
	}
	// construct the weighted L_E
	// start with unweighted values because they can be used in weight calculations
	vector<double> vechLwX;                                     // vech(L^w_X)
	vector<double> weights(Y_.getNcols(), 0.0);                 // will become a d-vector of weights (each element corresponding to a row of L_X; the first element is weighted T_E[1,1])
	for (size_t jCol = 0; jCol < Y_.getNcols() - 1; jCol++) {   // nothing to be done for the last column (it only has a diagonal element)
		for (size_t iRow = jCol + 1; iRow < Y_.getNcols(); iRow++) {
			double prod1 = Tx[jCol] * Le_.getElem(iRow, jCol);
			vechLwX.push_back(prod1);
			weights[iRow] += prod1 * Le_.getElem(iRow, jCol); // unweighted for now
		}
	}
	for (size_t k = 0; k < Y_.getNcols(); k++) {
		weights[k] = nu0_ * (weights[k] + Tx[k]) + invAsq_;
	}
	size_t vechInd = 0;
	for (size_t jCol = 0; jCol < Y_.getNcols() - 1; jCol++) {
		for (size_t iRow = jCol + 1; iRow < Y_.getNcols(); iRow++) {
			vechLwX[vechInd] = vechLwX[vechInd] / weights[iRow];
			vechInd++;
		}
	}
	// add the lower triangles and store the results in the gradient vector
	vechInd = 0;
	for (size_t jCol = 0; jCol < Y_.getNcols() - 1; jCol++) {
		for (size_t iRow = jCol + 1; iRow < Y_.getNcols(); iRow++) {
			grad[vechInd] = -mRtRLT.getElem(iRow, jCol) - nxnd_ * vechLwX[vechInd];
			vechInd++;
		}
	}
	// The T_E gradient
	// Starting with the first matrix: mRtRLT becomes L_E^TR^TRL_ET_E
	mRtRLT.trm('l', 'l', true, true, 1.0, Le_);
	// now sum everything and store the result in the gradient vector
	for (size_t k = 0; k < Y_.getNcols(); k++) {
		grad[fTeInd_ + k] = 0.5 * (Nnd_ - mRtRLT.getElem(k, k) - nxnd_ * Tx[k] / weights[k]);
	}
	// L_A and T_A next
	//
	// backtransform the logit-p_jp and scale
	vector<double> vP(Phi_.getNrows() * Phi_.getNcols(), 0.0);
	MatrixView P( &vP, 0, Phi_.getNrows(), Phi_.getNcols() );
	vector<double> pGrpSum(Phi_.getNrows(), 0.0);
	for (size_t m = 0; m < Phi_.getNcols(); m++) {
		for (size_t iRow = 0; iRow < Phi_.getNrows(); iRow++) {
			double p       = nuc_.logistic( Phi_.getElem(iRow, m) );
			pGrpSum[iRow] += p;
			P.setElem(iRow, m, p);
		}
	}
	for (size_t m = 0; m < Phi_.getNcols(); m++) {
		for (size_t iRow = 0; iRow < Phi_.getNrows(); iRow++) {
			double p = P.getElem(iRow, m) / pGrpSum[iRow];
			P.setElem( iRow, m, sqrt(p) ); // square root so that I can use syrk() below
		}
	}

	// Clear the Y residuals and re-use for A residuals
	vResid.clear();
	vResid.resize(A_.getNrows() * A_.getNcols(), 0.0);
	mResid = MatrixView( &vResid, 0, A_.getNrows(), A_.getNcols() );
	fill(vRtR.begin(), vRtR.end(), 0.0); // zero out vRtR, we will be adding the pop matrices to it
	for (size_t m = 0; m < Phi_.getNcols(); m++) {
		for (size_t jCol = 0; jCol  < A_.getNcols(); ++jCol) {
			for (size_t iRow = 0; iRow < A_.getNrows(); ++iRow) {
				double diff =  P.getElem(iRow, m) * ( A_.getElem(iRow, jCol) - Mp_.getElem(m, jCol) );
				mResid.setElem(iRow, jCol, diff); // sqrt(P[.p])(A - M[p.])
			}
		}
		mResid.syrk('l', 1.0, 1.0, mRtR); // adding to the mRtR that's from other groups
	}
	La_.symm('l', 'l', 1.0, mRtR, 0.0, mRtRLT); // R^TRL_A; R = A - Z_pM_p
	for (size_t k = 0; k < A_.getNcols(); k++) {
		Tx[k] = exp(viSig[fTaInd_ + k]);
	}
	// mutiply by T_A (the whole matrix because I will need it for left-multiplication later)
	for (size_t jCol = 0; jCol < A_.getNcols(); jCol++) {
		for (size_t iRow = 0; iRow < A_.getNcols(); iRow++) {
			double prod = mRtRLT.getElem(iRow, jCol) * Tx[jCol];
			mRtRLT.setElem(iRow, jCol, prod);
		}
	}
	weights.assign(weights.size(), 0.0);
	vechInd = 0;
	for (size_t jCol = 0; jCol < A_.getNcols() - 1; jCol++) {   // nothing to be done for the last column (it only has a diagonal element)
		for (size_t iRow = jCol + 1; iRow < A_.getNcols(); iRow++) {
			double prod1 = Tx[jCol] * La_.getElem(iRow, jCol);
			vechLwX[vechInd] = prod1;
			weights[iRow] += prod1 * La_.getElem(iRow, jCol); // unweighted for now
			vechInd++;
		}
	}
	for (size_t k = 0; k < A_.getNcols(); k++) {
		weights[k] = nu0_ * (weights[k] + Tx[k]) + invAsq_;
	}
	vechInd = 0;
	for (size_t jCol = 0; jCol < Y_.getNcols() - 1; jCol++) {
		for (size_t iRow = jCol + 1; iRow < Y_.getNcols(); iRow++) {
			vechLwX[vechInd] = vechLwX[vechInd] / weights[iRow];
			vechInd++;
		}
	}
	// add the lower triangles and store the results in the gradient vector
	vechInd = 0;
	for (size_t jCol = 0; jCol < Y_.getNcols() - 1; jCol++) {
		for (size_t iRow = jCol + 1; iRow < Y_.getNcols(); iRow++) {
			grad[fLaInd_+vechInd] = -mRtRLT.getElem(iRow, jCol) - nxnd_ * vechLwX[vechInd];
			vechInd++;
		}
	}
	// The T_A gradient
	// Starting with the first matrix: mRtRLT becomes L_A^TR^TRL_AT_A
	mRtRLT.trm('l', 'l', true, true, 1.0, La_);
	// now sum everything and store the result in the gradient vector
	for (size_t k = 0; k < A_.getNcols(); k++) {
		grad[fTaInd_ + k] = 0.5 * (NAnd_ - mRtRLT.getElem(k, k) - nxnd_ * Tx[k] / weights[k]);
	}
	// The T_G gradient
	// Start with calculating the residual, replacing the old one
	vResid.clear();
	for (size_t jCol = 0; jCol < Mp_.getNcols(); jCol++) {
		for (size_t iRow = 0; iRow < Mp_.getNrows(); iRow++) { // even if the group is empty; prior still has an effect
			vResid.push_back( Mp_.getElem(iRow, jCol) - mu_.getElem(0, jCol) );
		}
	}
	mResid = MatrixView( &vResid, 0, Mp_.getNrows(), Mp_.getNcols() );
	mResid.syrk('l', 1.0, 0.0, mRtR);
	for (size_t k = 0; k < A_.getNcols(); k++) {
		Tx[k] = exp(viSig[fTgInd_ + k]);
	}
	for (size_t k = 0; k < Mp_.getNcols(); k++) {
		weights[k] = nu0_ * Tx[k] + invAsq_;
	}
	// now sum everything and store the result in the gradient vector
	for (size_t k = 0; k < Mp_.getNcols(); k++) {
		grad[fTgInd_ + k] = 0.5 * (NGnd_ - mRtR.getElem(k, k) * Tx[k] - nxnd_ * Tx[k] / weights[k]);
	}
}

// WrapMMM methods

const double WrapMMM::lnMaxDbl_ = log( numeric_limits<double>::max() );

WrapMMM::WrapMMM(const vector<double> &vY, const size_t &d, const uint32_t &Ngrp, const double &alphaPr, const double &tau0, const double &nu0, const double &invAsq, vector<double> &testRes) : vY_{vY}, alphaPr_{alphaPr} {
#ifndef PKG_DEBUG_OFF
	if (vY_.size() % d) {
		throw string("WrapMMM no-replication constructor ERROR: length of response vector (") + to_string( vY_.size() ) + string(") not divisible by number of traits (") + to_string(d) + string(")");
	}
#endif
	const size_t N        = vY_.size() / d;
	const size_t grpDim   = Ngrp * d;
	const size_t dSq      = d * d;
	const size_t dd       = d * (d - 1) / 2;
	// Establish indexes and matrix views
	Y_ = MatrixView(&vY_, 0, N, d);
	vTheta_.resize(Ngrp * d + d + Ngrp * d * (d + 1) / 2 + d, 0.0);
	vLa_.resize(dSq * Ngrp, 0.0);
	La_.resize(Ngrp);
	size_t startLa = 0;
	firstLaInd_    = (Ngrp + 1) * d;
	for (auto &la : La_){
		la = MatrixView(&vLa_, startLa, d, d);
		startLa += dSq;
		for (size_t k = 0; k < d; k++) {
			la.setElem(k, k, 1.0);
		}
	}
	firstTaInd_ = firstLaInd_ + Ngrp * dd;

	// Obtain starting values using variational Bayes
	const double covRatio = 1.0;
	const int nReps       = 10;      // try this many VB fits, pick the best among them
	double dic            = 0.0;
	const double sig0     = 1.0;

	vector<double> vGrpMn;
	vector<double> vSm;
	vector<double> Nm;
	vector<double> r;
	vector<double> lPost;
	GmmVB vbModel(&vY_, covRatio, sig0, alphaPr_, Ngrp, d, &vGrpMn, &vSm, &r, &Nm);
	vbModel.fitModel(lPost, dic);
	for (int iRep = 1; iRep < nReps; iRep++) {
		vector<double> vGrpMnLoc;
		vector<double> rLoc;
		vector<double> vSmLoc;
		vector<double> NmLoc;
		vector<double> lPostLoc;
		double dicLoc = 0.0;

		GmmVB vbModel(&vY_, covRatio, sig0, alphaPr_, Ngrp, d, &vGrpMnLoc, &vSmLoc, &rLoc, &NmLoc);
		vbModel.fitModel(lPostLoc, dicLoc);
		if (dicLoc < dic) { // if we found a better DIC, save the important parameters
			vGrpMn = vGrpMnLoc;
			r      = rLoc;
			dic    = dicLoc;
		}
	}
	// Transfer the fitted location values to vTheta_
	Mp_ = MatrixView(&vTheta_, 0, Ngrp, d);
	memcpy( vTheta_.data(), vGrpMn.data(), grpDim * sizeof(double) );
	vector<double> tmpMu;
	Mp_.colMeans(tmpMu);
	memcpy( vTheta_.data() + grpDim, tmpMu.data(), d * sizeof(double) );
	MatrixView mu(&vTheta_, Ngrp * d, 1, d);
	MatrixView P(&r, 0, N, Ngrp);

	// Make factorized inverse-covariances using group means and responsibilities
	const double nInv = 1.0 / (static_cast<double>(N) - 1.0);
	for (size_t m = 0; m < Ngrp; m++) {
		vector<double> vRsd(vY_);
		MatrixView Rsd(&vRsd, 0, N, d);
		for (size_t jCol = 0; jCol < d; jCol++) {
			for (size_t iRow = 0; iRow < N; iRow++) {
				Rsd.subtractFromElem( iRow, jCol, Mp_.getElem(m, jCol) );
				Rsd.multiplyElem( iRow, jCol, P.getElem(iRow, m) );
			}
		}
		MatrixView S(&vSm, m * dSq, d, d);                               // re-using the covariance vector from the VB fit
		Rsd.syrk('l', nInv, 0.0, S);                                     // multiplying by 1/(N-1) gives the covariance in one shot
	}
	testRes = vSm;
	vTheta_[(Ngrp + 1) * d]      = -0.6;
	vTheta_[(Ngrp + 1) * d + 1]  =  0.6;
	vTheta_[(Ngrp + 1) * d + 2]  = -0.2;
	vTheta_[(Ngrp + 1) * d + 3]  =  0.307;
	vTheta_[(Ngrp + 1) * d + 4]  =  0.0;
	vTheta_[(Ngrp + 1) * d + 5]  =  0.307;
	vTheta_[(Ngrp + 1) * d + 6]  =  0.0;
	vTheta_[(Ngrp + 1) * d + 7]  =  0.039;
	vTheta_[(Ngrp + 1) * d + 8]  =  0.0;
	vTheta_[(Ngrp + 1) * d + 9]  = -4.6;
	vTheta_[(Ngrp + 1) * d + 10] = -4.6;

	vlnP_ = vector<double>(N * Ngrp, 0.0);
	lnP_  = MatrixView(&vlnP_, 0, N, Ngrp);
	const size_t grpSize = N / Ngrp;
	for (size_t m = 0; m < Ngrp; m++) {
		for (size_t iRow = 0; iRow < N; iRow++) {
			if ( (iRow >= m * grpSize) && (iRow <= (m + 1) * grpSize) ){
				lnP_.setElem( iRow, m, log(0.999) );
			} else {
				lnP_.setElem( iRow, m, log( 0.001 / static_cast<double>(Ngrp - 1) ) );
			}
		}
	}
	expandISvec_();
	lnPupdate_();
	models_.push_back( new MumiNR(&vY_, &vlnP_, d, Ngrp, tau0, nu0, invAsq) );
	samplers_.push_back( new SamplerNUTS(models_[0], &vTheta_) );
	//samplers_.push_back( new SamplerMetro(models_[0], &vTheta_, 0.05) );
}

WrapMMM::WrapMMM(const vector<double> &vY, const vector<size_t> &y2line, const uint32_t &Ngrp, const double &tauPrPhi, const double &alphaPr, const double &tau0, const double &nu0, const double &invAsq): vY_{vY}, alphaPr_{alphaPr}, hierInd_{Index(y2line)} {
	const size_t N = hierInd_.size();
#ifndef PKG_DEBUG_OFF
	if (vY.size()%N) {
		throw string("WrapMMM constructor ERROR: length of response vector not divisible by data point number");
	}
#endif
	const size_t d     = vY.size() / N;
	const size_t Nln   = hierInd_.groupNumber();
	const size_t Adim  = Nln * d;
	const size_t Mpdim = Ngrp * d;

	// Calculate starting values for theta
	Y_ = MatrixView(&vY_, 0, N, d);

	vTheta_.resize(Adim + Mpdim + d + Nln * Ngrp, 0.0);
	A_  = MatrixView(&vTheta_, 0, Nln, d);
	Mp_ = MatrixView(&vTheta_, Adim, Ngrp, d);
	MatrixView mu(&vTheta_, Adim+Mpdim, 1, d);

	Y_.colMeans(hierInd_[0], A_);  //  means to get A starting values
	vector<double> vSig(d * d, 0.0);
	MatrixView Sig(&vSig, 0, d, d);

	vector<size_t> ind;
	for (size_t m = 0; m < Ngrp; m++) {
		for (size_t iLn = 0; iLn < Nln / Ngrp; iLn++) {
			ind.push_back(m);
		}
	}
	Index popInd(ind);
	for (size_t m = 0; m < Ngrp; m++) {
		for (size_t iRow = 0; iRow < Nln; iRow++) {
			if (popInd.groupID(iRow) == m){
				lnP_.setElem( iRow, m, log(0.99) );
			} else {
				lnP_.setElem( iRow, m, log(0.01) );
			}
		}
	}
	A_.colMeans(popInd, Mp_);
	// use k-means for group assignment and starting values of logit(p_jp)
	//Index popInd(Ngrp);
	//vector<double> vtM(Mpdim, 0.0);
	//MatrixView tmpM(&vtM, 0, Ngrp, d);
	/*
	kMeans_(A_, Ngrp, 50, popInd, Mp_);
	//kMeans_(A_, Ngrp, 50, popInd, tmpM);
	std::fstream tstInd;
	tstInd.open("yIndTst.txt", std::ios::trunc | std::ios::out);
	vector<size_t> ind;
	for (size_t m = 0; m < Ngrp; m++) {
		ind.push_back(popInd[m][0]);
		tstInd << popInd[m][0] << " " << std::flush;
	}
	tstInd << std::endl;
	vector<size_t> trkInd;
	insertionSort(ind, trkInd);
	for (auto &i : trkInd){
		tstInd << i << " " << std::flush;
	}
	tstInd << std::endl;
	tstInd.close();
	for (size_t m = 0; m < Ngrp; m++) {
		for (size_t iRow = 0; iRow < Nln; iRow++) {
			if (popInd.groupID(iRow) == m){
				Phi_.setElem( iRow, m, logit( 0.8 + 0.15 * rng_.runif() ) );
			} else {
				Phi_.setElem( iRow, m, logit( 0.1 + 0.15 * rng_.runif() ) );
			}
		}
	}
	*/
	vector<double> tmpMu;
	Mp_.colMeans(tmpMu);
	for (size_t k = 0; k < d; k++) {
		mu.setElem(0, k, tmpMu[k]);
	}

	// Calculate starting precision matrix values; do that before adding noise to theta
	//
	size_t trLen     = d * (d-1) / 2;
	fTeInd_          = trLen;
	//fLaInd_          = trLen + d;
	//fTaInd_          = fLaInd_ + trLen;
	const double n   = 1.0 / static_cast<double>(N-1);
	const double nLN = 1.0 / static_cast<double>(Nln-1);
	const double nP  = static_cast<double>(Ngrp-1); // not reciprocal on purpose
	vLa_.resize(d * d, 0.0);
	//La_ = MatrixView(&vLa_, 0, d, d);

	// Y residual
	vector<double> vZA(N * d, 0.0);
	MatrixView ZA(&vZA, 0, N, d);
	A_.colExpand(hierInd_[0], ZA);
	for (size_t jCol = 0; jCol < d; jCol++) {
		for (size_t iRow = 0; iRow < N; iRow++) {
			double diff = Y_.getElem(iRow, jCol) - ZA.getElem(iRow, jCol);
			ZA.setElem(iRow, jCol, diff); // ZA now Y - ZA
		}
	}
	ZA.syrk('l', n, 0.0, Sig); // making covariances in one step
	Sig.pseudoInv();

	// save the scaled precision matrix lower triangle and log-diagonals to the precision parameter vector
	vector<double> sqrT;
	for (size_t k = 0; k < d; k++) {
		sqrT.push_back( sqrt( Sig.getElem(k, k) ) );
	}
	// A precision matrix
	vZA.resize(Nln * d);
	MatrixView ZpMp(&vZA, 0, Nln, d);
	Mp_.colExpand(popInd, ZpMp);
	for (size_t jCol = 0; jCol < d ; jCol++) {
		for (size_t iRow = 0; iRow < Nln ; iRow++) {
			double diff = A_.getElem(iRow, jCol) - ZpMp.getElem(iRow, jCol);
			ZpMp.setElem(iRow, jCol, diff); // A - ZM is now in ZM
		}
	}
	ZpMp.syrk('l', nLN, 0.0, Sig);
	Sig.pseudoInv();

	for (size_t k = 0; k < d; k++) {
		sqrT[k] = sqrt( Sig.getElem(k, k) );
	}
	expandISvec_();
	vAresid_.resize(Adim, 0.0);
	Aresid_ = MatrixView(&vAresid_, 0, Nln, d);
	sortGrps_();
	// add noise
	/*
	for (size_t iTht = 0; iTht < PhiBegInd_; iTht++) { // the Phi values already have added noise
		vTheta_[iTht] += 0.5 * rng_.rnorm();
	}
	for (auto &s : vISig_) {
		s += 0.5 * rng_.rnorm();
	}
	*/
	samplers_.push_back( new SamplerNUTS(models_[0], &vTheta_) );
	//samplers_.push_back( new SamplerMetro(models_[0], &vTheta_) );
	//samplers_.push_back( new SamplerMetro(models_[1], &vISig_) );
}

WrapMMM::WrapMMM(const vector<double> &vY, const vector<size_t> &y2line, const vector<int32_t> &missIDs, const uint32_t &Ngrp, const double &tauPrPhi, const double &alphaPr, const double &tau0, const double &nu0, const double &invAsq) : WrapMMM(vY, y2line, Ngrp, tauPrPhi, alphaPr, tau0, nu0, invAsq) {
	for (size_t jCol = 0; jCol < A_.getNcols(); jCol++) {
		for (size_t iRow = 0; iRow < hierInd_[0].size(); iRow++) {
			if (missIDs[jCol * hierInd_[0].size() + iRow]) {
				missInd_[iRow].push_back(jCol); // if the iRow element does not yet exist, it will be created
			}
		}
	}
	imputeMissing_();
}

WrapMMM::~WrapMMM() {
	for (auto &m : models_) {
		delete m;
	}
	for (auto &s : samplers_) {
		delete s;
	}
}

void WrapMMM::lnPupdate_() {
	// start by summing group assignment probabilities
	const size_t Ngrp = lnP_.getNcols();
	const size_t N    = lnP_.getNrows();
	const size_t d    = Y_.getNcols();
	vector<double> alpha(Ngrp, 0.0);                                           // will become beta distribution concentrations
	for (size_t m = 0; m < Ngrp; m++) {
		for (size_t iRow = 0; iRow < N;	iRow++) {
			alpha[m] += exp( lnP_.getElem(iRow, m) );
		}
	}
	for (auto &a : alpha){
		a += alphaPr_;
	}
	vector<double> phi(Ngrp, 0.0);
	rng_.rdirichlet(alpha, phi);
	// set up the matrix of group kernels
	vector<double> vKappa(N * Ngrp, 0.0);
	MatrixView Kappa( &vKappa, 0, N, Ngrp );
	size_t tInd = firstTaInd_;
	vector<double> lambda(Ngrp, 0.0);                                          // Sigma_a determinant (lambda in the model document)
	for (size_t m = 0; m < Ngrp; m++) {                                        // m is the group index as in the model description document
		vector<double> vResid(vY_);                                            // copy over Y_
		MatrixView mResid = MatrixView(&vResid, 0, N, d);                      // mResid now has the Y values
		for (size_t jCol = 0; jCol < d; jCol++) {
			for (size_t iRow = 0; iRow < N; iRow++) {
				mResid.subtractFromElem( iRow, jCol, Mp_.getElem(m, jCol) );   // mResid now Y - mu_m
			}
		}
		mResid.trm('l', 'r', false, true, 1.0, La_[m]);                        // mResid now (Y-mu_m)L_A
		for (size_t jCol = 0; jCol < d; jCol++) {
			for (size_t iRow = 0; iRow < N; iRow++) {
				double rsd = mResid.getElem(iRow, jCol);
				Kappa.addToElem(iRow, m, exp(vTheta_[tInd]) * rsd * rsd);      // (Y-mu_m)L_A T_A L_A^T(Y - mu_g)^T
			}
			lambda[m] += vTheta_[tInd];
			tInd++;
		}
		for (size_t iRow = 0; iRow < N; iRow++) {
			const double kappaNew = log(phi[m]) + 0.5 * ( lambda[m] - Kappa.getElem(iRow, m) );
			Kappa.setElem(iRow, m, kappaNew);
		}
	}
	// Now calculate the new lnP_
	for (size_t m = 0; m < Ngrp; m++) {
		for (size_t iRow = 0; iRow < N; iRow++) {
			const double curKappa = Kappa.getElem(iRow, m);
			bool noOverflow  = true;
			double sumOfExp  = 0.0;
			for (size_t mKappa = 0; mKappa < Ngrp; mKappa++) {
				if (mKappa == m){
					continue;
				} else {
					double diff = Kappa.getElem(iRow, mKappa) - curKappa;
					if (diff >= lnMaxDbl_){ // exponentiation will overflow; just set the result to -lnMaxDbl_ as an approximation
						noOverflow = false;
						lnP_.setElem(iRow, m, -lnMaxDbl_);
						break;
					} else {
						diff = exp(diff);
						if ( (numeric_limits<double>::max() - sumOfExp) <= diff ){ // adding the next value will overflow the sum
							noOverflow = false;
							lnP_.setElem(iRow, m, -lnMaxDbl_);
							break;
						} else {
							sumOfExp += diff;
						}
					}
				}
			}
			if (noOverflow){
				lnP_.setElem(iRow, m, -log1p(sumOfExp));
			}
		}
	}
}

void WrapMMM::imputeMissing_() {
	if ( missInd_.size() ) { // impute only of there are missing data
		// start by making Sigma_e^{-1}
		vector<double> Te;
		// convert log-tau to tau
		/*
		for (size_t p = fTeInd_; p < fLaInd_; p++) {
			Te.push_back( exp(vISig_[p]) );
		}
		*/
		vector<double> vSigI(A_.getNcols() * A_.getNcols(), 0.0);
		MatrixView SigI( &vSigI, 0, A_.getNcols(), A_.getNcols() );
		// fill out the matrix; the Le lower triangle is the first [0,fTeInd_) elements on vISig_
		// there may be room for optimization here: we are working by row, while the matrix is stored by column, precluding vectorization
		// first row and column are trivial because l_11 == 1.0 and all other row elements are 0.0
		/*  TODO: fix this once the rest of the model is done
		vSigI[0] = Te[0];
		for (size_t i = 1; i < SigI.getNcols(); i++) {
			double prd = Te[0] * vISig_[i-1];
			SigI.setElem(i, 0, prd); // only need the lower triangle for chol()
		}
		size_t dPr = SigI.getNcols() - 1;
		for (size_t jCol = 1; jCol < SigI.getNcols(); jCol++) {
			double diagVal = Te[jCol];
			size_t ind     = jCol - 1;
			for (size_t k = 0; k < jCol; k++) {
				diagVal += Te[k] * vISig_[ind] * vISig_[ind];
				ind     += dPr - k - 1;
			}
			SigI.setElem(jCol, jCol, diagVal);
			for (size_t iRow = jCol + 1; iRow < SigI.getNrows(); iRow++) {
				double val  = 0.0;
				size_t cInd = jCol - 1; // the column (L^T) index
				size_t rInd = iRow - 1; // the row (L) index
				for (size_t k = 0; k < jCol; k++) { // stopping at jCol because that is the shorter non-0 run
					val  += Te[k] * vISig_[rInd] * vISig_[cInd];
					cInd += dPr - k - 1;
					rInd += dPr - k - 1;
				}
				val += Te[jCol] * vISig_[rInd];
				SigI.setElem(iRow, jCol, val); // only need the lower triangle for chol()
			}
		}
		*/
		vector<double> vSig(vSigI.size(), 0.0);
		MatrixView Sig( &vSig, 0, SigI.getNrows(), SigI.getNcols() );
		// Invert SigI
		//SigI.chol(Sig);
		//Sig.cholInv();
		SigI.pseudoInv(Sig);
		// go through all rows of Y_ with missing data
		for (auto &missRow : missInd_) {
			// make Sig_aa^-1 (corresponding to the absent traits)
			vector<double> vSigAA( missRow.second.size() * missRow.second.size() );
			MatrixView SigAA( &vSigAA, 0, missRow.second.size(), missRow.second.size() );
			for (size_t jNew = 0; jNew < missRow.second.size(); jNew++) {
				for (size_t iNew = jNew; iNew < missRow.second.size(); iNew++) { // only need the lower triangle for chol()
					SigAA.setElem( iNew, jNew, SigI.getElem(missRow.second[iNew], missRow.second[jNew]) );
				}
			}
			SigAA.pseudoInv(); // [(Sig^-1)_aa]^-1; Schur complement of Sig_pp
			SigAA.chol();    // this will be the covariance for the MV Gaussian
			// subset Sig for mean calculation
			size_t dP = Sig.getNcols() - missRow.second.size(); // number of present traits
			vector<double> vSigPP;
			vector<double> vSigAP;
			size_t missIDcol = 0;
			for (size_t jCol = 0; jCol < Sig.getNcols(); jCol++) {
				if ( ( missIDcol < missRow.second.size() ) && (jCol == missRow.second[missIDcol]) ) { // skip any column corresponding to a missing trait
					missIDcol++;
					continue;
				}
				size_t missIDrow = 0;
				for (size_t iRow = 0; iRow < Sig.getNrows(); iRow++) {
					if ( ( missIDrow < missRow.second.size() ) && (iRow == missRow.second[missIDrow]) ) { // row with missing data
						vSigAP.push_back( Sig.getElem(iRow, jCol) );
						missIDrow++;
					} else {
						vSigPP.push_back( Sig.getElem(iRow, jCol) );
					}
				}
			}
			MatrixView SigPP(&vSigPP, 0, dP, dP);
			SigPP.pseudoInv(); // now Sig_pp^-1
			MatrixView SigAP(&vSigAP, 0, missRow.second.size(), dP);
			// generate mean present and absent vectors
			vector<double> muA;     // mu_a (means corresponding to the missing data)
			vector<double> diffMuP; // a - mu_g (difference vector corresponding to the present values)
			missIDcol = 0;
			for (size_t k = 0; k < Y_.getNcols(); k++) {
				if ( ( missIDcol < missRow.second.size() ) && (k == missRow.second[missIDcol]) ) {
					muA.push_back( A_.getElem(hierInd_.groupID(missRow.first), k) );
					missIDcol++;
				} else {
					diffMuP.push_back( Y_.getElem(missRow.first, k) - A_.getElem(hierInd_.groupID(missRow.first), k) );
				}
			}
			MatrixView muAmat(&muA, 0, muA.size(), 1);
			MatrixView AMdiff(&diffMuP, 0, diffMuP.size(), 1);
			vector<double> vSigProd(SigAP.getNrows() * SigPP.getNcols(), 0.0);
			MatrixView SigProd( &vSigProd, 0, SigAP.getNrows(), SigPP.getNcols() );
			SigAP.symm('l', 'r', 1.0, SigPP, 0.0, SigProd); // Sig_apSig_pp^-1
			AMdiff.gemm(false, 1.0, SigProd, false, 1.0, muAmat);
			// sample from a multivariate normal
			vector<double> z;
			for (size_t k = 0; k < missRow.second.size(); k++) {
				z.push_back( rng_.rnorm() );
			}
			MatrixView zMat(&z, 0, missRow.second.size(), 1);
			zMat.trm('l', 'l', false, false, 0.0, SigAA); // z now scaled by chol(Sig_aa)
			missIDcol = 0;
			for (auto &k : missRow.second) {
				Y_.setElem(missRow.first, k, muA[missIDcol] + z[missIDcol]);
				missIDcol++;
			}
		}
	}
}

void WrapMMM::sortGrps_() {
	vector<size_t> firstIdx;                                       // vector with indices of the first high-p elements per group
	for (size_t m = 0; m < lnP_.getNcols(); m++) {
		for (size_t iRow = 0; iRow < lnP_.getNrows(); iRow++) {
			if (lnP_.getElem(iRow, m) >= -0.05129329){ // ln(0.95)
				firstIdx.push_back(iRow);
				break;
			}
		}
	}
	if ( firstIdx.size() < lnP_.getNcols() ){                      // some groups may have no high-probability individuals
		while ( firstIdx.size() != lnP_.getNcols() ){
			firstIdx.push_back( lnP_.getNrows() );                 // add one past the last index, to guarantee that these groups will be put last
		}
	}
	vector<size_t> popIdx;
	//nuc_.insertionSort(firstIdx, popIdx);
	Mp_.permuteRows(popIdx);
	lnP_.permuteCols(popIdx);
}

void WrapMMM::expandISvec_() {
	const size_t d = Y_.getNcols();
	size_t aInd = firstLaInd_;                                     // index of the La lower triangle in the input vector
	for (size_t m = 0; m < Mp_.getNrows(); m++){
		for (size_t jCol = 0; jCol < d - 1; jCol++) {              // the last column is all 0, except the last element = 1.0
			for (size_t iRow = jCol + 1; iRow < d; iRow++) {
				La_[m].setElem(iRow, jCol, vTheta_[aInd]);
				aInd++;
			}
		}
	}
}


double WrapMMM::rowDistance_(const MatrixView &m1, const size_t &row1, const MatrixView &m2, const size_t &row2) {
#ifndef PKG_DEBUG_OFF
	if ( m1.getNcols() != m2.getNcols() ) {
		throw string("ERROR: m1 and m2 matrices must have the same number of columns in WrapMMM::rowDist_()");
	}
	if ( row1+1 > m1.getNrows() ) {
		throw string("ERROR: row1  index out of bounds in WrapMMM::rowDist_()");
	}
	if ( row2+1 > m2.getNrows() ) {
		throw string("ERROR: row2  index out of bounds in WrapMMM::rowDist_()");
	}
#endif
	double dist = 0.0;
	for (size_t jCol = 0; jCol < m1.getNcols(); jCol++) {
		double diff = m1.getElem(row1, jCol) - m2.getElem(row2, jCol);
		dist += diff * diff;
	}
	return sqrt(dist);
}

void WrapMMM::kMeans_(const MatrixView &X, const size_t &Kclust, const uint32_t &maxIt, Index &x2m, MatrixView &M) {
#ifndef PKG_DEBUG_OFF
	if (M.getNrows() != Kclust) {
		throw string("ERROR: Matrix of means must have one row per cluster in WrapMMM::kMeans_()");
	}
	if ( X.getNcols() != M.getNcols() ) {
		throw string("ERROR: Matrix of observations must have the same number of columns as the matrix of means in WrapMMM::kMeans_()");
	}
	if ( M.getNrows() != x2m.groupNumber() ) {
		throw string("ERROR: observation to cluster index must be the same number of groups as the number of groups in WrapMMM::kMeans_()");
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
		if ( ( nDiff / static_cast<double>( X.getNrows() ) ) <= 0.1 ) { // fewer than 10% of assignments changed
			break;
		}
	}
}

void WrapMMM::runSampler(const uint32_t &Nadapt, const uint32_t &Nsample, const uint32_t &Nthin, vector<double> &thetaChain, vector<double> &piChain) {
	thetaChain.clear();
	piChain.clear();
	for (uint32_t a = 0; a < Nadapt; a++) {
		size_t parGrp = 0;
		for (auto &s : samplers_) {
			int16_t tr = s->adapt();
			parGrp++;
		}
		expandISvec_();
		lnPupdate_();
	}
	for (uint32_t b = 0; b < Nsample; b++) {
		size_t parGrp = 0;
		for (auto &s : samplers_) {
			int16_t tr = s->update();
			parGrp++;
		}
		expandISvec_();
		lnPupdate_();
		if ( (b%Nthin) == 0) {
			for (auto &t : vTheta_){
				thetaChain.push_back(t);
			}
			for (auto &p : vlnP_){
				piChain.push_back( exp(p) );
			}
		}
	}
}

void WrapMMM::runSampler(const uint32_t &Nadapt, const uint32_t &Nsample, const uint32_t &Nthin, vector<double> &thetaChain, vector<double> &isigChain, vector<double> &piChain, vector<double> &impYchain) {
	for (uint32_t a = 0; a < Nadapt; a++) {
		for (auto &s : samplers_) {
			s->adapt();
		}
		//sortGrps_();
		imputeMissing_();
	}
	for (uint32_t b = 0; b < Nsample; b++) {
		for (auto &s : samplers_) {
			s->update();
		}
		//sortGrps_();
		imputeMissing_();
		if ( (b%Nthin) == 0) {
			/*
			for (size_t iTht = 0; iTht < PhiBegInd_; iTht++) {
				thetaChain.push_back(vTheta_[iTht]);
			}
			for (size_t jCol = 0; jCol < Phi_.getNcols(); jCol++) {
				for (size_t iRow = 0; iRow < Phi_.getNrows(); iRow++) {
					piChain.push_back( nuc_.logistic( Phi_.getElem(iRow, jCol) ) );
				}
			}
			for (auto &p : vISig_) {
				isigChain.push_back(p);
			}
			for (auto &m : missInd_) {
				for (auto &k : m.second) {
					impYchain.push_back( Y_.getElem(m.first, k) );
				}
			}
			*/
		}
	}
}


