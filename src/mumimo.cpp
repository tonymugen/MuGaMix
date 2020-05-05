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

#include <fstream>

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
/** \brief Swap two `size_t` values
 *
 * Uses the three XORs trick to swap two integers. Safe if the variables happen to refer to the same address.
 *
 * \param[in,out] i first integer
 * \param[in,out] j second integer
 */
void swapXOR(size_t &i, size_t &j){
	if (&i != &j) { // no move needed if this is actually the same variable
		i ^= j;
		j ^= i;
		i ^= j;
	}
}
/** \brief Insertion sort
 *
 * Performs an insertion sort on a vector, outputting the position of each element in a sorted vector. Sorting is done in order of increase.
 *
 * \param[in] vec vector to be sorted
 * \param[out] ind vector of sorted indexes, will replace contents of non-empty vector
 */
void insertionSort(const vector<size_t> &vec, vector<size_t> &ind){
	if ( ind.size() ){
		ind.clear();
	}

	for (size_t l = 0; l < vec.size(); l++) {
		ind.push_back(l);
	}
	for (size_t i = 1; i < vec.size(); i++) {
		size_t j   = i;
		size_t tmp = ind[i];
		while ( (j > 0) && (vec[ ind[j-1] ] > vec[tmp]) ){
			//swapXOR(ind[j], ind[j-1]);
			ind[j] = ind[j-1];
			j--;
		}
		ind[j] = tmp;
	}
}
/** \brief Logit function
 *
 * \param[in] p probability in the (0, 1) interval
 * \return logit transformation
 */
inline double logit(const double &p){ return log(p) - log(1.0 - p); }

/** \brief Logistic function
 *
 * There is a guard against under- and overflow: the function returns 0.0 for \f$ x \le -35.0\f$ and 1.0 for \f$x \ge 35.0\f$.
 *
 * \param[in] x value to be projected to the (0, 1) interval
 * \return logistic transformation
 */
double logistic(const double &x){
	// 35.0 is the magic number because logistic(-35) ~ EPS
	// the other cut-offs have been empirically determined
	if (x <= - 35.0){
		return 0.0;
	} else if (x >= 35.0){
		return 1.0;
	} else if (x <= -7.0){ // approximation for smallish x
		return exp(x);
	} else if (x >= 3.5){  // approximation for largish x
		return 1.0 - exp(-x);
	} else {
		return 1.0/(1 + exp(-x));
	}
}

/** \brief Shell sort
 *
 * Sorts the provided vector in ascending order using Shell's method. Rather than move the elements themselves, save their indexes to the output vector. The first element of the index vector points to the smallest element of the input vector etc. The implementation is modified from code in Numerical Recipes in C++.
 * NOTE: This algorithm is too slow for vectors of \f$ > 50\f$ elements. I am using it for population projection ordering, where the number of populations will typically not exceed 10.
 *
 * \param[in] target vector to be sorted
 * \param[out] outIdx vector of indexes
 */
void sort(const vector<double> &target, vector<size_t> &outIdx){
	if ( outIdx.size() ) {
		outIdx.clear();
	}
	for (size_t i = 0; i < target.size(); i++) {
		outIdx.push_back(i);
	}
	// pick the initial increment
	size_t inc = 1;
	do {
		inc = inc*3 + 1;
	} while ( inc <= target.size() );

	// start the sort
	do { // loop over partial sorts, decreasing the increment each time
		inc /= 3;
		const size_t bottom = inc;
		for (size_t iOuter = bottom; iOuter < target.size(); iOuter++) { // outer loop of the insertion sort, going over the indexes
#ifndef PKG_DEBUG_OFF
			if ( outIdx[iOuter] >= target.size() ) {
				throw string("outIdx value out of bounds for target vector in shellSort()");
			}
#endif
			const size_t curInd = outIdx[iOuter]; // save the current value of the index
			size_t jInner       = iOuter;
			while (target[ outIdx[jInner - inc] ] > target[curInd]) {  // Straight insertion inner loop; looking for a place to insert the current value
#ifndef PKG_DEBUG_OFF
				if ( outIdx[jInner-inc] >= target.size() ) {
					throw string("outIdx value out of bounds for target vector in shellSort()");
				}
#endif
				outIdx[jInner] = outIdx[jInner-inc];
				jInner        -= inc;
				if (jInner < bottom) {
					break;
				}
			}
			outIdx[jInner] = curInd;
		}
	} while (inc > 1);
}

// MumiLoc methods
const double MumiLoc::pSumCutOff_ = 0.003;

MumiLoc::MumiLoc(const vector<double> *yVec, const vector<double> *iSigVec, const vector<Index> *hierInd, const double &tau, const size_t &nPops, const double &tauPrPhi) : Model(), hierInd_{hierInd}, tau0_{tau}, iSigTheta_{iSigVec}, Npop_{nPops}, tauPrPhi_{tauPrPhi} {
	const size_t n = (*hierInd_)[0].size();
#ifndef PKG_DEBUG_OFF
	if (yVec->size()%n) {
		throw string("ERROR: Y dimensions not compatible with the number of data points implied by the replicate factor");
	}
#endif
	const size_t d  = yVec->size()/n;
	Y_              = MatrixViewConst(yVec, 0, n, d);

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
	PhiBegInd_   = ( (*hierInd_)[0].groupNumber() + Npop_ + 1 )*d;
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
		Npop_      = in.Npop_;
		tauPrPhi_ = in.tauPrPhi_;

		in.hierInd_   = nullptr;
		in.iSigTheta_ = nullptr;
	}
}

MumiLoc& MumiLoc::operator=(MumiLoc &&in){
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
		Npop_      = in.Npop_;
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
	const size_t Ydim = Y_.getNrows()*Y_.getNcols();
	MatrixViewConst A( &theta, 0, Nln, Y_.getNcols() );
	MatrixViewConst Mp( &theta, Nln*Y_.getNcols(), Npop_, Y_.getNcols() );
	MatrixViewConst mu( &theta, (Nln+Npop_)*Y_.getNcols(), 1, Y_.getNcols() ); // overall mean
	MatrixViewConst Phi( &theta, PhiBegInd_, Nln, Npop_ );

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
			dp += mResid.getElem(iRow, jCol)*mResid.getElem(iRow, jCol);
		}
		eTrace   += exp( (*iSigTheta_)[fTeInd_ + jCol] )*dp;
	}

	// backtransform the logit-p_jp, calculate row sums and sum of squares
	vector<double> vP(Phi.getNrows()*Phi.getNcols(), 0.0);
	MatrixView P( &vP, 0, Phi.getNrows(), Phi.getNcols() );
	vector<double> pPopSum(Nln, 0.0);
	double sumPhiSq = 0.0;
	for (size_t m = 0; m < Phi.getNcols(); m++) {
		for (size_t iRow = 0; iRow < Phi.getNrows(); iRow++) {
			double phi     = Phi.getElem(iRow, m);
			sumPhiSq      += phi*phi;
			double p       = logistic(phi);
			pPopSum[iRow] += p;
			P.setElem(iRow, m, p);
		}
	}
	// Re-weight the P; it may be possible to optimize this further by dividing by weight once after aTrace completion
	for (size_t m = 0; m < Phi.getNcols(); m++) {
		for (size_t iRow = 0; iRow < Phi.getNrows(); iRow++) {
			if (pPopSum[iRow] <= pSumCutOff_){  // approximation when all p_j are small
				double aSum = 0.0;
				double phiM = Phi.getElem(iRow, m);
				for (size_t jCol = 0; jCol < Phi.getNcols(); jCol++) {
					if (jCol == m){
						aSum += 1.0;
					} else {
						aSum += exp(Phi.getElem(iRow, jCol) - phiM);
					}
					P.setElem(iRow, m, 1.0/aSum);
				}
			} else {
				P.divideElem(iRow, m, pPopSum[iRow]);
			}
		}
	}
	// Clear the Y residuals and re-use for A residuals
	vResid.clear();
	vector<double> AtraceVec(Nln, 0.0); // accumulate A trace values here
	// calculate T_A
	vector<double> Ta;
	for (size_t k = fTaInd_; k < fTpInd_; k++) {
		Ta.push_back( exp( (*iSigTheta_)[k] ) );
	}
	for (size_t m = 0; m < Npop_; m++) {                                           // m is the population index as in the model description document
		vector<double> locAtr(Nln, 0.0);
		vResid.assign( theta.begin(), theta.begin() + A.getNrows()*A.getNcols() ); // copy over A
		mResid = MatrixView( &vResid, 0, A.getNrows(), A.getNcols() );             // mResid now has the A values
		for (size_t jCol = 0; jCol < A.getNcols(); jCol++) {
			for (size_t iRow = 0; iRow < A.getNrows(); iRow++) {
				mResid.subtractFromElem(iRow, jCol, Mp.getElem(m, jCol));          // mResid now A - mu_m
			}
		}
		mResid.trm('l', 'r', false, true, 1.0, La_);                               // mResid now (A-mu_m)L_A
		for (size_t jCol = 0; jCol < A.getNcols(); jCol++) {
			for (size_t iRow = 0; iRow < A.getNrows(); iRow++) {
				double rsd    = mResid.getElem(iRow, jCol);
				locAtr[iRow] += Ta[jCol]*rsd*rsd;                                  // (A-mu_m)L_A T_A L_A^T(A - mu_p)^T
			}
		}
		for (size_t j = 0; j < Nln; j++) {
			AtraceVec[j] += P.getElem(j, m)*locAtr[j];                             // P_m(A-mu_m)L_A T_A L_A^T(A-mu_m)^T
		}
	}
	double aTrace = 0.0;
	for (auto &a : AtraceVec){
		aTrace += a;
	}
	// M[p] crossproduct trace
	double trM = 0.0;
	for (size_t jCol = 0; jCol < Mp.getNcols(); ++jCol) {
		double dp = 0.0;
		for (size_t iRow = 0; iRow < Mp.getNrows(); ++iRow) {
			double diff = Mp.getElem(iRow, jCol) - mu.getElem(0, jCol);
			dp += diff*diff;
		}
		trM += exp( (*iSigTheta_)[fTpInd_ + jCol] )*dp;
	}
	double trP = 0.0;
	for (size_t jCol = 0; jCol < Y_.getNcols(); jCol++) {
		trP += mu.getElem(0, jCol)*mu.getElem(0, jCol);
	}
	trP *= tau0_;
	// now sum to get the log-posterior
	return -0.5*(eTrace + aTrace + tauPrPhi_*sumPhiSq + trM + trP);
}

void MumiLoc::gradient(const vector<double> &theta, vector<double> &grad) const{
	expandISvec_();
	if ( grad.size() ) {
		grad.clear();
	}
	grad.resize(theta.size(), 0.0);
	const size_t Nln  = (*hierInd_)[0].groupNumber();
	const size_t Ydim = Y_.getNrows()*Y_.getNcols();
	const size_t Adim = Nln*Y_.getNcols();
	const size_t Mdim = Npop_*Y_.getNcols();
	MatrixViewConst A( &theta, 0, Nln, Y_.getNcols() );
	MatrixViewConst M( &theta, Adim, Npop_, Y_.getNcols() );
	MatrixViewConst mu( &theta, Adim + Mdim, 1, Y_.getNcols() ); // overall mean
	MatrixViewConst Phi(&theta, PhiBegInd_, Nln, Npop_);

	// Matrix views of the gradient
	MatrixView gA( &grad, 0, Nln, Y_.getNcols() );
	MatrixView gM( &grad, Adim, Npop_, Y_.getNcols() );
	MatrixView gmu( &grad, Adim + Mdim, 1, Y_.getNcols() ); // overall mean
	MatrixView gPhi(&grad, PhiBegInd_, Nln, Npop_);

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
	vector<double> vISigE(Y_.getNcols()*Y_.getNcols(), 0.0);
	MatrixView iSigE( &vISigE, 0, Y_.getNcols(), Y_.getNcols() );
	// L_ExT_E
	vector<double> Tx;
	for (size_t k = fTeInd_; k < fLaInd_; k++) {
		Tx.push_back( exp( (*iSigTheta_)[k] ) );
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
	MatrixView iSigA( &vISigA, 0, Y_.getNcols(), Y_.getNcols() );
	for (size_t k = 0; k < iSigA.getNcols(); k++) {
		Tx[k] = exp( (*iSigTheta_)[k + fTaInd_] );
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
	MatrixView mResISE( &vResISE, 0, Y_.getNrows(), Y_.getNcols() );
	mResid.symm('l', 'r', 1.0, iSigE, 0.0, mResISE);

	// backtransform the logit-p_jp and calculate row sums
	vector<double> vP(Phi.getNrows()*Phi.getNcols());
	MatrixView P( &vP, 0, Phi.getNrows(), Phi.getNcols() );
	vector<double> pPopSum(Nln, 0.0);
	for (size_t m = 0; m < Phi.getNcols(); m++) {
		for (size_t iRow = 0; iRow < Phi.getNrows(); iRow++) {
			double p       = logistic(Phi.getElem(iRow, m));
			pPopSum[iRow] += p;
			P.setElem(iRow, m, p);
		}
	}
	// Re-weight p_jm = p_jm/sum_m(p_jm)
	for (size_t m = 0; m < Phi.getNcols(); m++) {
		for (size_t iRow = 0; iRow < Phi.getNrows(); iRow++) {
			if (pPopSum[iRow] <= pSumCutOff_){  // approximation when all p_j are small
				double aSum = 0.0;
				double phiM = Phi.getElem(iRow, m);
				for (size_t jCol = 0; jCol < Phi.getNcols(); jCol++) {
					if (jCol == m){
						aSum += 1.0;
					} else {
						aSum += exp(Phi.getElem(iRow, jCol) - phiM);
					}
					P.setElem(iRow, m, 1.0/aSum);
				}
			} else {
				P.divideElem(iRow, m, pPopSum[iRow]);
			}
		}
	}
	// Z^T(Y - ZA - XB)Sig[E]^-1 store in gA
	mResISE.colSums((*hierInd_)[0], gA);
	vector<double> kernSum(P.getNrows(), 0.0);
	for (size_t m = 0; m < Npop_; m++) {
		vector<double> vAresid(theta.begin(), theta.begin()+Adim);                                  // copying A to the residual matrix
		MatrixView Aresid( &vAresid, 0, A.getNrows(), A.getNcols() );
		for (size_t jCol = 0; jCol < A.getNcols(); jCol++) {
			for (size_t iRow = 0; iRow < A.getNrows(); iRow++) {
				Aresid.subtractFromElem(iRow, jCol, M.getElem(m, jCol));                            // A - mu_m
			}
		}
		vector<double> vAresISA(A.getNrows()*A.getNcols(), 0.0);
		MatrixView AresISA( &vAresISA, 0, A.getNrows(), A.getNcols() );
		Aresid.symm('l', 'r', 1.0, iSigA, 0.0, AresISA);                                            // (A - mu_m)Sigma^{-1}_A

		// accumulate appropriately weighted (a_j - mu_m)Sigma^{-1}_A(a_j - mu_m)^T in gPhi
		for (size_t jCol = 0; jCol < A.getNcols(); jCol++) {
			for (size_t iRow = 0; iRow < A.getNrows(); iRow++) {
				gPhi.addToElem(iRow, m, Aresid.getElem(iRow, jCol)*AresISA.getElem(iRow, jCol));    // calculate the (j,m) kernel and store in gPhi
			}
		}
		for (size_t iRow = 0; iRow < A.getNrows(); iRow++) {
			kernSum[iRow] += P.getElem(iRow, m)*gPhi.getElem(iRow, m);                              // add the kernels together
		}

		for (size_t jCol = 0; jCol < A.getNcols(); jCol++) {
			for (size_t iRow = 0; iRow < A.getNrows(); iRow++) {
				AresISA.multiplyElem(iRow, jCol, P.getElem(iRow, m));                               // P_m(A - mu_m)Sigma^{-1}_A
			}
		}
		// finish A partial derivatives
		for (size_t jCol = 0; jCol < A.getNcols(); jCol++) {
			for (size_t iRow = 0; iRow < A.getNrows(); iRow++) {
				gA.subtractFromElem(iRow, jCol, AresISA.getElem(iRow, jCol));                       // subtracting each population's P_m(A - mu_m)Sigma^{-1}_A
			}
		}
		// sum the A residuals into gM rows
		for (size_t jCol = 0; jCol < A.getNcols(); jCol++) {
			for (size_t iRow = 0; iRow < A.getNrows(); iRow++) {
				gM.addToElem(m, jCol, AresISA.getElem(iRow, jCol));
			}
		}
	}
	// kernel_m - sum(kernel_l)
	for (size_t m = 0; m < Npop_; m++) {
		for (size_t iRow = 0; iRow < P.getNrows(); iRow++) {
			gPhi.subtractFromElem(iRow, m, kernSum[iRow]);
		}
	}
	// M partial derivatives
	// make tau_p
	vector<double> tauP;
	for (size_t k = fTpInd_; k < iSigTheta_->size(); k++) {
		tauP.push_back( exp( (*iSigTheta_)[k] ) );
	}
	// Population mean residual
	vector<double> vPOPresid(theta.begin()+Adim, theta.begin()+Adim+Mdim);
	MatrixView POPresid( &vPOPresid, 0, M.getNrows(), M.getNcols() );
	for (size_t jCol = 0; jCol < POPresid.getNcols(); jCol++) {
		for (size_t iRow = 0; iRow < POPresid.getNrows(); iRow++) {
			POPresid.subtractFromElem( iRow, jCol, mu.getElem(0, jCol) );
			POPresid.multiplyElem(iRow, jCol, tauP[jCol]); // (M-mu)T_M
		}
	}
	// complete the M gradient
	for (size_t jCol = 0; jCol < M.getNcols(); jCol++) {
		for (size_t iRow = 0; iRow < M.getNrows(); iRow++) {
			gM.subtractFromElem( iRow, jCol, POPresid.getElem(iRow, jCol) );  // gM already has the summed A residuals
		}
	}
	// mu partial derivatives
	vector <double> PresSum; // colSums will resize
	POPresid.colSums(PresSum);
	for (size_t jCol = 0; jCol < M.getNcols(); jCol++) {
		PresSum[jCol] -= mu.getElem(0, jCol)*tau0_;
	}
	for (size_t jCol = 0; jCol < gmu.getNcols(); jCol++) {
		gmu.setElem(0, jCol, PresSum[jCol]);
	}
	// Phi partial derivatives
	for (size_t m = 0; m < Npop_; m++) {
		for (size_t iRow = 0; iRow < gPhi.getNrows(); iRow++) {
			double phi  = -0.5*gPhi.getElem(iRow, m);  // gPhi already has the weighted kernel product sums stored from before; t_A element missing because we have one Sigma_A
			double p    = P.getElem(iRow, m);
			phi        *= p*(1.0 - p);
			gPhi.setElem(iRow, m, phi - tauPrPhi_*Phi.getElem(iRow, m));
		}
	}
}

// MumiISig methods
MumiISig::MumiISig(const vector<double> *yVec, const vector<double> *vTheta, const vector<Index> *hierInd, const double &nu0, const double &invAsq, const size_t &nPops) : Model(), hierInd_{hierInd}, nu0_{nu0}, invAsq_{invAsq} {
	const size_t N = (*hierInd_)[0].size(); // first index is data to lines
#ifndef PKG_DEBUG_OFF
	if (yVec->size()%N) {
		throw string("MumiISig constructor ERROR: vectorized data length not divisible by number of data points");
	}
#endif
	const size_t d = yVec->size()/N;
#ifndef PKG_DEBUG_OFF
	if (vTheta->size()%d) {
		throw string("MumiISig constructor ERROR: vectorized parameter set length not divisible by number of traits");
	}
#endif
	Y_ = MatrixViewConst(yVec, 0, N, d);
	const size_t Nln = (*hierInd_)[0].groupNumber();
	A_   = MatrixViewConst(vTheta, 0, Nln, d);
	Mp_  = MatrixViewConst(vTheta, Nln*d, nPops, d);
	mu_  = MatrixViewConst(vTheta, (Nln+nPops)*d, 1, d);
	Phi_ = MatrixViewConst(vTheta, (Nln+nPops+1)*d, Nln, nPops);
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
	nxnd_        = nu0_*( nu0_ + 2.0*static_cast<double>(d) );
	Nnd_         = static_cast<double>( Y_.getNrows() ) + nu0_ + 2.0*static_cast<double>(d);
	NAnd_        = static_cast<double>( A_.getNrows() ) + nu0_ + 2.0*static_cast<double>(d);
	NPnd_        = static_cast<double>(nPops) + nu0_ + 2.0*static_cast<double>(d);
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
		La_      = MatrixView( &vLx_, Y_.getNcols()*Y_.getNcols(), Y_.getNcols(), Y_.getNcols() );
		fTeInd_  = in.fTeInd_;
		fLaInd_  = in.fLaInd_;
		fTaInd_  = in.fTaInd_;
		nxnd_    = in.nxnd_;
		Nnd_     = in.Nnd_;
		NAnd_    = in.NAnd_;
		NPnd_    = in.NPnd_;

		in.hierInd_ = nullptr;
	}
}

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
		Phi_     = move(in.Phi_);
		vLx_     = move(in.vLx_);
		Le_      = MatrixView( &vLx_, 0, Y_.getNcols(), Y_.getNcols() );
		La_      = MatrixView( &vLx_, Y_.getNcols()*Y_.getNcols(), Y_.getNcols(), Y_.getNcols() );
		fTeInd_  = in.fTeInd_;
		fLaInd_  = in.fLaInd_;
		fTaInd_  = in.fTaInd_;
		fTpInd_  = in.fTpInd_;
		nxnd_    = in.nxnd_;
		Nnd_     = in.Nnd_;
		NAnd_    = in.NAnd_;
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
			dp += mResid.getElem(iRow, jCol)*mResid.getElem(iRow, jCol);
		}
		eTrace += exp(viSig[fTeInd_ + jCol])*dp;
	}

	// backtransform the logit-p_jp and sum
	vector<double> vP(Phi_.getNrows()*Phi_.getNcols(), 0.0);
	MatrixView P( &vP, 0, Phi_.getNrows(), Phi_.getNcols() );
	vector<double> pPopSum(Phi_.getNrows(), 0.0);
	for (size_t m = 0; m < Phi_.getNcols(); m++) {
		for (size_t iRow = 0; iRow < Phi_.getNrows(); iRow++) {
			double p       = logistic( Phi_.getElem(iRow, m) );
			pPopSum[iRow] += p;
			P.setElem(iRow, m, p);
		}
	}
	for (size_t m = 0; m < Phi_.getNcols(); m++) {
		for (size_t iRow = 0; iRow < Phi_.getNrows(); iRow++) {
			P.divideElem(iRow, m, pPopSum[iRow]);
		}
	}

	// Clear the Y residuals and re-use for A residuals
	vResid.clear();
	vResid.resize(A_.getNrows()*A_.getNcols(), 0.0);
	mResid = MatrixView( &vResid, 0, A_.getNrows(), A_.getNcols() );
	vector<double> AtraceVec(A_.getNrows(), 0.0); // accumulate A trace values here
	// calculate T_A
	vector<double> Ta;
	for (size_t k = fTaInd_; k < fTpInd_; k++) {
		Ta.push_back( exp(viSig[k]) );
	}
	for (size_t m = 0; m < Phi_.getNcols(); m++) {                                 // m is the population index as in the model description document
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
				locAtr[iRow] += Ta[jCol]*rsd*rsd;                                  // (A-mu_m)L_A T_A L_A^T(A - mu_p)^T
			}
		}
		for (size_t j = 0; j < A_.getNrows(); j++) {
			AtraceVec[j] += P.getElem(j, m)*locAtr[j];                             // P_m(A-mu_m)L_A T_A L_A^T(A-mu_m)^T
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
			dp += diff*diff;
		}
		trM += exp(viSig[fTpInd_ + jCol])*dp;
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
	ldetSumE *= Nnd_;
	ldetSumA *= NAnd_;
	ldetSumP *= NPnd_;
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
	pTrace *= nu0_ + 2.0*static_cast<double>( Y_.getNcols() );
	return -0.5*(eTrace + aTrace + trM - ldetSumE - ldetSumA - ldetSumP + pTrace);
}

void MumiISig::gradient(const vector<double> &viSig, vector<double> &grad) const{
	// expand the element vector to make the L matrices
	expandISvec_(viSig);
	if ( grad.size() ){
		grad.clear();
	}
	grad.resize(viSig.size(), 0.0);
	// Calculate the Y residuals
	vector<double> vResid(Y_.getNcols()*Y_.getNrows(), 0.0);
	MatrixView mResid( &vResid, 0, Y_.getNrows(), Y_.getNcols() );
	for (size_t jCol = 0; jCol  < Y_.getNcols(); ++jCol) {
		for (size_t iRow = 0; iRow < Y_.getNrows(); ++ iRow) {
			double diff =  Y_.getElem(iRow, jCol) - A_.getElem( (*hierInd_)[0].groupID(iRow), jCol );
			mResid.setElem(iRow, jCol, diff); // Y - ZA
		}
	}
	vector<double> vRtR(Y_.getNcols()*Y_.getNcols(), 0.0);
	MatrixView mRtR( &vRtR, 0, Y_.getNcols(), Y_.getNcols() );
	mResid.syrk('l', 1.0, 0.0, mRtR);
	vector<double> vRtRLT(Y_.getNcols()*Y_.getNcols(), 0.0);
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
	// backtransform the logit-p_jp and scale
	vector<double> vP(Phi_.getNrows()*Phi_.getNcols(), 0.0);
	MatrixView P( &vP, 0, Phi_.getNrows(), Phi_.getNcols() );
	vector<double> pPopSum(Phi_.getNrows(), 0.0);
	for (size_t m = 0; m < Phi_.getNcols(); m++) {
		for (size_t iRow = 0; iRow < Phi_.getNrows(); iRow++) {
			double p       = logistic( Phi_.getElem(iRow, m) );
			pPopSum[iRow] += p;
			P.setElem(iRow, m, p);
		}
	}
	for (size_t m = 0; m < Phi_.getNcols(); m++) {
		for (size_t iRow = 0; iRow < Phi_.getNrows(); iRow++) {
			double p = P.getElem(iRow, m)/pPopSum[iRow];
			P.setElem(iRow, m, sqrt(p)); // square root so that I can use syrk() below
		}
	}

	// Clear the Y residuals and re-use for A residuals
	vResid.clear();
	vResid.resize(A_.getNrows()*A_.getNcols(), 0.0);
	mResid = MatrixView( &vResid, 0, A_.getNrows(), A_.getNcols() );
	fill(vRtR.begin(), vRtR.end(), 0.0); // zero out vRtR, we will be adding the pop matrices to it
	for (size_t m = 0; m < Phi_.getNcols(); m++) {
		for (size_t jCol = 0; jCol  < A_.getNcols(); ++jCol) {
			for (size_t iRow = 0; iRow < A_.getNrows(); ++iRow) {
				double diff =  P.getElem(iRow, m)*( A_.getElem(iRow, jCol) - Mp_.getElem(m, jCol) );
				mResid.setElem(iRow, jCol, diff); // sqrt(P[.p])(A - M[p.])
			}
		}
		mResid.syrk('l', 1.0, 1.0, mRtR); // adding to the mRtR that's from other populations
	}
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
	// Start with calculating the residual, replacing the old one
	vResid.clear();
	for (size_t jCol = 0; jCol < Mp_.getNcols(); jCol++) {
		for (size_t iRow = 0; iRow < Mp_.getNrows(); iRow++) { // even if the population is empty; prior still has an effect
			vResid.push_back( Mp_.getElem(iRow, jCol) - mu_.getElem(0, jCol) );
		}
	}
	mResid = MatrixView( &vResid, 0, Mp_.getNrows(), Mp_.getNcols() );
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
const double WrapMMM::phiMin_ = -5.0;
const double WrapMMM::addVal_ = 3.0;

WrapMMM::WrapMMM(const vector<double> &vY, const vector<size_t> &y2line, const uint32_t &Npop, const double &alphaPr, const double &tau0, const double &nu0, const double &invAsq): vY_{vY} {
	hierInd_.push_back( Index(y2line) );
	const size_t N = hierInd_[0].size();
#ifndef PKG_DEBUG_OFF
	if (vY.size()%N) {
		throw string("WrapMMM constructor ERROR: length of response vector not divisible by data point number");
	}
#endif
	const size_t d     = vY.size()/N;
	const size_t Nln   = hierInd_[0].groupNumber();
	const size_t Adim  = Nln*d;
	const size_t Mpdim = Npop*d;

	// Calculate starting values for theta
	Y_ = MatrixView(&vY_, 0, N, d);

	vTheta_.resize(Adim + Mpdim + d + Nln*Npop, 0.0);
	A_  = MatrixView(&vTheta_, 0, Nln, d);
	Mp_ = MatrixView(&vTheta_, Adim, Npop, d);
	MatrixView mu(&vTheta_, Adim+Mpdim, 1, d);
	PhiBegInd_ = Adim+Mpdim+d;
	Phi_       = MatrixView(&vTheta_, PhiBegInd_, Nln, Npop);

	Y_.colMeans(hierInd_[0], A_);  //  means to get A starting values
	vector<double> vSig(d*d, 0.0);
	MatrixView Sig(&vSig, 0, d, d);

	vector<size_t> ind;
	for (size_t m = 0; m < Npop; m++) {
		for (size_t iLn = 0; iLn < Nln/Npop; iLn++) {
			ind.push_back(m);
		}
	}
	Index popInd(ind);
	for (size_t m = 0; m < Npop; m++) {
		for (size_t iRow = 0; iRow < Nln; iRow++) {
			if (popInd.groupID(iRow) == m){
				Phi_.setElem( iRow, m, logit(0.99) );
			} else {
				Phi_.setElem( iRow, m, logit(0.01) );
			}
		}
	}
	A_.colMeans(popInd, Mp_);
	// use k-means for population assignment and starting values of logit(p_jp)
	//Index popInd(Npop);
	//vector<double> vtM(Mpdim, 0.0);
	//MatrixView tmpM(&vtM, 0, Npop, d);
	/*
	kMeans_(A_, Npop, 50, popInd, Mp_);
	//kMeans_(A_, Npop, 50, popInd, tmpM);
	std::fstream tstInd;
	tstInd.open("yIndTst.txt", std::ios::trunc | std::ios::out);
	vector<size_t> ind;
	for (size_t m = 0; m < Npop; m++) {
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
	for (size_t m = 0; m < Npop; m++) {
		for (size_t iRow = 0; iRow < Nln; iRow++) {
			if (popInd.groupID(iRow) == m){
				Phi_.setElem( iRow, m, logit( 0.8 + 0.15*rng_.runif() ) );
			} else {
				Phi_.setElem( iRow, m, logit( 0.1 + 0.15*rng_.runif() ) );
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
	size_t trLen     = d*(d-1)/2;
	fTeInd_          = trLen;
	fLaInd_          = trLen + d;
	fTaInd_          = fLaInd_ + trLen;
	const double n   = 1.0/static_cast<double>(N-1);
	const double nLN = 1.0/static_cast<double>(Nln-1);
	const double nP  = static_cast<double>(Npop-1); // not reciprocal on purpose
	vLa_.resize(d*d, 0.0);
	La_ = MatrixView(&vLa_, 0, d, d);

	// Y residual
	vector<double> vZA(N*d, 0.0);
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
		sqrT.push_back( sqrt(Sig.getElem(k, k)) );
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
	sortPops_();
	// add noise
	/*
	for (size_t iTht = 0; iTht < PhiBegInd_; iTht++) { // the Phi values already have added noise
		vTheta_[iTht] += 0.5*rng_.rnorm();
	}
	for (auto &s : vISig_) {
		s += 0.5*rng_.rnorm();
	}
	*/
	models_.push_back( new MumiLoc(&vY_, &vISig_, &hierInd_, tau0, Npop, alphaPr) );
	models_.push_back( new MumiISig(&vY_, &vTheta_, &hierInd_, nu0, invAsq, Npop) );
	samplers_.push_back( new SamplerNUTS(models_[0], &vTheta_) );
	//samplers_.push_back( new SamplerMetro(models_[0], &vTheta_) );
	samplers_.push_back( new SamplerNUTS(models_[1], &vISig_) );
	//samplers_.push_back( new SamplerMetro(models_[1], &vISig_) );
}

WrapMMM::WrapMMM(const vector<double> &vY, const vector<size_t> &y2line, const vector<int32_t> &missIDs, const uint32_t &Npop, const double &alphaPr, const double &tau0, const double &nu0, const double &invAsq) : WrapMMM(vY, y2line, Npop, alphaPr, tau0, nu0, invAsq) {
	for (size_t jCol = 0; jCol < A_.getNcols(); jCol++) {
		for (size_t iRow = 0; iRow < hierInd_[0].size(); iRow++) {
			if (missIDs[jCol*hierInd_[0].size() + iRow]) {
				missInd_[iRow].push_back(jCol); // if the iRow element does not yet exist, it will be created
			}
		}
	}
	imputeMissing_();
}

WrapMMM::~WrapMMM(){
	for (auto &m : models_) {
		delete m;
	}
	for (auto &s : samplers_) {
		delete s;
	}
}

void WrapMMM::imputeMissing_(){
	if ( missInd_.size() ) { // impute only of there are missing data
		// start by making Sigma_e^{-1}
		vector<double> Te;
		// convert log-tau to tau
		for (size_t p = fTeInd_; p < fLaInd_; p++) {
			Te.push_back( exp(vISig_[p]) );
		}
		vector<double> vSigI(A_.getNcols()*A_.getNcols(), 0.0);
		MatrixView SigI( &vSigI, 0, A_.getNcols(), A_.getNcols() );
		// fill out the matrix; the Le lower triangle is the first [0,fTeInd_) elements on vISig_
		// there may be room for optimization here: we are working by row, while the matrix is stored by column, precluding vectorization
		// first row and column are trivial because l_11 == 1.0 and all other row elements are 0.0
		vSigI[0] = Te[0];
		for (size_t i = 1; i < SigI.getNcols(); i++) {
			double prd = Te[0]*vISig_[i-1];
			SigI.setElem(i, 0, prd); // only need the lower triangle for chol()
		}
		size_t dPr = SigI.getNcols() - 1;
		for (size_t jCol = 1; jCol < SigI.getNcols(); jCol++) {
			double diagVal = Te[jCol];
			size_t ind     = jCol - 1;
			for (size_t k = 0; k < jCol; k++) {
				diagVal += Te[k]*vISig_[ind]*vISig_[ind];
				ind     += dPr - k - 1;
			}
			SigI.setElem(jCol, jCol, diagVal);
			for (size_t iRow = jCol + 1; iRow < SigI.getNrows(); iRow++) {
				double val  = 0.0;
				size_t cInd = jCol - 1; // the column (L^T) index
				size_t rInd = iRow - 1; // the row (L) index
				for (size_t k = 0; k < jCol; k++) { // stopping at jCol because that is the shorter non-0 run
					val  += Te[k]*vISig_[rInd]*vISig_[cInd];
					cInd += dPr - k - 1;
					rInd += dPr - k - 1;
				}
				val += Te[jCol]*vISig_[rInd];
				SigI.setElem(iRow, jCol, val); // only need the lower triangle for chol()
			}
		}
		vector<double> vSig(vSigI.size(), 0.0);
		MatrixView Sig( &vSig, 0, SigI.getNrows(), SigI.getNcols() );
		// Invert SigI
		//SigI.chol(Sig);
		//Sig.cholInv();
		SigI.pseudoInv(Sig);
		// go through all rows of Y_ with missing data
		for (auto &missRow : missInd_) {
			//make Sig_aa^-1 (corresponding to the absent traits)
			vector<double> vSigAA( missRow.second.size()*missRow.second.size() );
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
			vector<double> diffMuP; // a - mu_p (difference vector corresponding to the present values)
			missIDcol = 0;
			for (size_t k = 0; k < Y_.getNcols(); k++) {
				if ( ( missIDcol < missRow.second.size() ) && (k == missRow.second[missIDcol]) ) {
					muA.push_back( A_.getElem(hierInd_[0].groupID(missRow.first), k) );
					missIDcol++;
				} else {
					diffMuP.push_back( Y_.getElem(missRow.first, k) - A_.getElem(hierInd_[0].groupID(missRow.first), k) );
				}
			}
			MatrixView muAmat(&muA, 0, muA.size(), 1);
			MatrixView AMdiff(&diffMuP, 0, diffMuP.size(), 1);
			vector<double> vSigProd(SigAP.getNrows()*SigPP.getNcols(), 0.0);
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

void WrapMMM::sortPops_(){}

void WrapMMM::calibratePhi_(){
	std::fstream tstRecal;
	tstRecal.open("tstRecal.txt", std::ios::app);
	for (size_t iRow = 0; iRow < Phi_.getNrows(); iRow++) {
		if (Phi_.getElem(iRow, 0) <= phiMin_){
			uint8_t doCalibrate = 0;
			for (size_t jCol = 1; jCol < Phi_.getNcols(); jCol++) {
				if (Phi_.getElem(iRow, jCol) > phiMin_){            // any large enough element lets us bail
					doCalibrate = 0;
					break;
				}
				doCalibrate = 1;
			}
			if (doCalibrate){
				tstRecal << iRow << ": " << std::flush;
				for (size_t jCol = 0; jCol < Phi_.getNcols(); jCol++) {
					tstRecal << Phi_.getElem(iRow, jCol) << " " << std::flush;
					Phi_.addToElem(iRow, jCol, addVal_);
				}
				tstRecal << std::endl;
			}
		}
	}
	tstRecal << "-----------" << std::endl;
	tstRecal.close();
}

double WrapMMM::rowDistance_(const MatrixView &m1, const size_t &row1, const MatrixView &m2, const size_t &row2){
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
		dist += diff*diff;
	}
	return sqrt(dist);
}

void WrapMMM::kMeans_(const MatrixView &X, const size_t &Kclust, const uint32_t &maxIt, Index &x2m, MatrixView &M){
#ifndef PKG_DEBUG_OFF
	if (M.getNrows() != Kclust) {
		throw string("ERROR: Matrix of means must have one row per cluster in WrapMMM::kMeans_()");
	}
	if ( X.getNcols() != M.getNcols() ) {
		throw string("ERROR: Matrix of oservations must have the same number of cloumns as the matrix of means in WrapMMM::kMeans_()");
	}
	if ( M.getNrows() != x2m.groupNumber() ) {
		throw string("ERROR: observation to cluster index must be the same number of groups as the number of populations in WrapMMM::kMeans_()");
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
	vector<size_t> sNew(X.getNrows(), 0); // new cluster assigment vector
	for (uint32_t i = 0; i < maxIt; i++) {
		// save the previous S vector
		sPrevious = sNew;
		// assign cluster IDs according to minimal diistance
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
		if ( (nDiff/static_cast<double>( X.getNrows() )) <= 0.1 ) { // fewer than 10% of assignments changed
			break;
		}
	}
}

void WrapMMM::runSampler(const uint32_t &Nadapt, const uint32_t &Nsample, const uint32_t &Nthin, vector<double> &thetaChain, vector<double> &isigChain, vector<double> &piChain){
	std::fstream treeOut;
	treeOut.open("treeTests.tsv", std::ios::trunc | std::ios::out);
	treeOut << "y\tvariable.group\tphase" << std::endl;
	for (uint32_t a = 0; a < Nadapt; a++) {
		size_t parGrp = 0;
		for (auto &s : samplers_) {
			size_t tr = s->adapt();
			treeOut << tr << "\t" << parGrp << "\tadapt" << std::endl;
			parGrp++;
		}
		calibratePhi_();
		//sortPops_();
	}
	for (uint32_t b = 0; b < Nsample; b++) {
		size_t parGrp = 0;
		for (auto &s : samplers_) {
			size_t tr = s->update();
			treeOut << tr << "\t" << parGrp << "\tsample" << std::endl;
			parGrp++;
		}
		calibratePhi_();
		//sortPops_();
		if ( (b%Nthin) == 0) {
			for (size_t iTht = 0; iTht < PhiBegInd_; iTht++) {
				thetaChain.push_back(vTheta_[iTht]);
			}
			vector<double> vW;
			for (size_t jCol = 0; jCol < Phi_.getNcols(); jCol++) {
				for (size_t iRow = 0; iRow < Phi_.getNrows(); iRow++) {
					vW.push_back( logistic(Phi_.getElem(iRow, jCol)) );
				}
			}
			MatrixView W( &vW, 0, Phi_.getNrows(), Phi_.getNcols() );
			vector<double> rowSums;
			W.rowSums(rowSums);
			for (size_t jCol = 0; jCol < Phi_.getNcols(); jCol++) {
				for (size_t iRow = 0; iRow < Phi_.getNrows(); iRow++) {
					piChain.push_back( W.getElem(iRow, jCol)/rowSums[iRow] );
				}
			}
			for (auto &p : vISig_) {
				isigChain.push_back(p);
			}
		}
	}
	treeOut.close();
}

void WrapMMM::runSampler(const uint32_t &Nadapt, const uint32_t &Nsample, const uint32_t &Nthin, vector<double> &thetaChain, vector<double> &isigChain, vector<double> &piChain, vector<double> &impYchain){
	for (uint32_t a = 0; a < Nadapt; a++) {
		for (auto &s : samplers_) {
			s->adapt();
		}
		//sortPops_();
		imputeMissing_();
	}
	for (uint32_t b = 0; b < Nsample; b++) {
		for (auto &s : samplers_) {
			s->update();
		}
		//sortPops_();
		imputeMissing_();
		if ( (b%Nthin) == 0) {
			for (size_t iTht = 0; iTht < PhiBegInd_; iTht++) {
				thetaChain.push_back(vTheta_[iTht]);
			}
			for (size_t jCol = 0; jCol < Phi_.getNcols(); jCol++) {
				for (size_t iRow = 0; iRow < Phi_.getNrows(); iRow++) {
					piChain.push_back( logistic(Phi_.getElem(iRow, jCol)) );
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
		}
	}
}


