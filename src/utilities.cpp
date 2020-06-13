/*
 * Copyright (c) <YEAR> Anthony J. Greenberg
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

/// Numerical utilities implementation
/** \file
 * \author Anthony J. Greenberg
 * \copyright Copyright (c) 2020 Anthony J. Greenberg
 * \version 1.0
 *
 * Class implementation for a set of numerical utilities.
 * Implemented as a class because this seems to be the only way for these methods to be included using Rcpp with no compilation errors.
 *
 */

#include <vector>
#include <cmath>
#include <string>
#include <limits>

#include "utilities.hpp"
#include "matrixView.hpp"

using namespace BayesicSpace;
using std::vector;
using std::string;
using std::to_string;
using std::numeric_limits;

const double NumerUtil::gCoeff_[14] {
	57.1562356658629235,     59.5979603554754912,
	14.1360979747417471,     -0.491913816097620199,
	0.339946499848118887e-4,  0.465236289270485756e-4,
	-0.983744753048795646e-4, 0.158088703224912494e-3,
	-0.210264441724104883e-3, 0.217439618115212643e-3,
	-0.164318106536763890e-3, 0.844182239838527433e-4,
	-0.261908384015814087e-4, 0.368991826595316234e-5
};
const double NumerUtil::bvalues_[22] = {
	1.00000000000000000e+00, -5.00000000000000000e-01,
	1.66666666666666667e-01, -3.33333333333333333e-02,
	2.38095238095238095e-02, -3.33333333333333333e-02,
	7.57575757575757576e-02, -2.53113553113553114e-01,
	1.16666666666666667e+00, -7.09215686274509804e+00,
	5.49711779448621554e+01, -5.29124242424242424e+02,
	6.19212318840579710e+03, -8.65802531135531136e+04,
	1.42551716666666667e+06, -2.72982310678160920e+07,
	6.01580873900642368e+08, -1.51163157670921569e+10,
	4.29614643061166667e+11, -1.37116552050883328e+13,
	4.88332318973593167e+14, -1.92965793419400681e+16
};

void NumerUtil::swapXOR(size_t &i, size_t &j) const{
	if (&i != &j) { // no move needed if this is actually the same variable
		i ^= j;
		j ^= i;
		i ^= j;
	}
}
void NumerUtil::insertionSort(const vector<size_t> &vec, vector<size_t> &ind) const{
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
			ind[j] = ind[j-1];
			j--;
		}
		ind[j] = tmp;
	}
}
double NumerUtil::logistic(const double &x) const{
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
		return 1.0/( 1 + exp(-x) );
	}
}
void NumerUtil::sort(const vector<double> &target, vector<size_t> &outIdx) const{
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
double NumerUtil::lnGamma(const double &x) const{
	if (x <= 0.0) return nan("");

	// define the weird magical coefficients
	// save a copy of x for incrementing
	double y     = x;
	double gamma = 5.24218750000000000; // 671/128
	double tmp   = x + gamma;
	tmp          = (x + 0.5)*log(tmp) - tmp;
	double logPi = 0.91893853320467267;  // 0.5*log(2.0*pi)
	tmp         += logPi;
	double cZero = 0.999999999999997092; // c_0

	for (size_t i = 0; i < 14; i++) {
		cZero += gCoeff_[i]/(++y);
	}

	return tmp + log(cZero/x);
}
double NumerUtil::digamma(const double &x) const{
	const int32_t nMax = 100;
	if (x <= 0.0){
		return nan("");
	}
	if ( isnan(x) ){
		return x;
	}
	// very large x
    double xln = log(x);
	double lrg = 1/( 2.0*numeric_limits<double>::epsilon() );
	if(x * xln > lrg) {
		return xln;
	}
	const int32_t n    = (-numeric_limits<double>::min_exponent < numeric_limits<double>::max_exponent ? -numeric_limits<double>::min_exponent : numeric_limits<double>::max_exponent);
	const double r1m4  = 0.5*numeric_limits<double>::epsilon();
	const double r1m5  = 0.301029995663981195213738894724;            // log_10(2)
	const double wdtol = (r1m4 > 0.5e-18 ? r1m4 : 0.5e-18);
    const double elim  = 2.302*(static_cast<double>(n)*r1m5 - 3.0);  // = 700.6174...
	// small x and underflow conditions
	if (xln < -elim){
		return nan(""); // underflow
	} else if (x < wdtol){
		return -1.0/x;
	}

	// regular calculations
	double rln   = r1m5*static_cast<double>(numeric_limits<double>::digits);
	rln          = (rln < 18.06 ? rln : 18.06);
	double fln   = (rln > 3.0 ? rln-3.0 : 0.0);
	if (fln < 0.0){
		throw string("ERROR: fln value ") + to_string(fln) + string(" less than 0 in locDigamma()");
	}
	const double fn   = 3.50 + 0.40*fln;
	const double xmin = ceil(fn);
	double xdmy       = x;
	double xdmln      = xln;
	double xinc       = 0.0;
	if (x < xmin) {
		xinc  = xmin - floor(x);
		xdmy  = x + xinc;
		xdmln = log(xdmy);
	}

	double tk     = 2.0*xdmln;
	if (tk <= elim) { // for x not large
		double t1   = 0.5/xdmy;
		double tst  = wdtol*t1;
		double rxsq = 1.0/(xdmy*xdmy);
		double t    = 0.5*rxsq;
		double s    = t*bvalues_[2];
		if (fabs(s) >= tst) {
			tk = 2.0;
			for(uint16_t k = 4; k <= 22; k++) {
				t         *= ( (tk + 1.0)/(tk + 1.0) )*( tk/(tk + 2.0) )*rxsq;
				double tmp = t * bvalues_[k-1];
				if (fabs(tmp) < tst) {
					break;
				}
				s += tmp;
				tk += 2.0;
			}
		}
		s += t1;
		if (xinc > 0.0) {
			// backward recursion from xdmy to x
			int32_t nx = static_cast<int32_t>(xinc);
			if (nx > nMax) {
				throw string("Increment ") + to_string(nx) + string(" too large in locDigamma()");
			}
			for(int32_t i = 1; i <= nx; i++){
				s += 1.0/( x + static_cast<double>(nx - i) ); // avoid disastrous cancellation, according to the comment in the R code
			}
		}
		return xdmln - s;
	} else {
		double s   = -x;
		double den = x;
		for(uint32_t i=0; i < static_cast<uint32_t>(fln) + 1; i++) { // checked fln for < 0.0, so this should be safe
			den += 1.0;
			s   += 1.0/den;
		}
		return -s;
	}
}
void NumerUtil::phi2p(const MatrixViewConst &Phi, MatrixView &P) const{
#ifndef PKG_DEBUG_OFF
	if ( ( Phi.getNcols()+1 != P.getNcols() ) || ( Phi.getNrows() != P.getNrows() ) ){
		throw string("ERROR: Phi (") + to_string( Phi.getNrows() ) + "x" + to_string( Phi.getNcols() ) +  string(") and P (") + to_string( P.getNrows() ) + "x" + to_string( P.getNcols() ) + string(") dimensions incompatible in phi2p(MatrixViewConst &, MatrixView &)");
	}
#endif
	for (size_t m = 0; m < Phi.getNcols(); m++) {
		for (size_t iRow = 0; iRow < Phi.getNrows(); iRow++) {
			P.setElem( iRow, m, logistic( Phi.getElem(iRow, m) ) );
		}
	}
	// Re-weight the P using the Betancourt (2012) algorithm
	vector<double> rowProd(Phi.getNrows(), 0.0);
	for (size_t m = 0; m < P.getNcols(); m++) {
		for (size_t iRow = 0; iRow < P.getNrows(); iRow++) {
			if (m == 0){
				rowProd[iRow] = P.getElem(iRow, m);
				P.setElem(iRow, m, 1.0 - rowProd[iRow]);
			} else if ( m == Phi.getNcols() ){           // Phi has one fewer columns than P
				P.setElem(iRow, m, rowProd[iRow]);
			} else {
				double psi = P.getElem(iRow, m);
				P.setElem( iRow, m, rowProd[iRow]*(1.0 - psi) );
				rowProd[iRow] *= psi;
			}
		}
	}
}
void NumerUtil::w2p(const MatrixViewConst &W, MatrixView &P) const{
#ifndef PKG_DEBUG_OFF
	if ( ( W.getNcols()+1 != P.getNcols() ) || ( W.getNrows() != P.getNrows() ) ){
		throw string("ERROR: W (") + to_string( W.getNrows() ) + "x" + to_string( W.getNcols() ) +  string(") and P (") + to_string( P.getNrows() ) + "x" + to_string( P.getNcols() ) + string(") dimensions incompatible in w2p(MatrixViewConst &, MatrixView &)");
	}
#endif
	// Re-weight the P using the Betancourt (2012) algorithm
	vector<double> rowProd(W.getNrows(), 0.0);
	for (size_t m = 0; m < P.getNcols(); m++) {
		for (size_t iRow = 0; iRow < P.getNrows(); iRow++) {
			if (m == 0){
				rowProd[iRow] = W.getElem(iRow, m);
				P.setElem(iRow, m, 1.0 - rowProd[iRow]);
			} else if ( m == W.getNcols() ){
				P.setElem(iRow, m, rowProd[iRow]);
			} else {
				double psi = W.getElem(iRow, m);
				P.setElem( iRow, m, rowProd[iRow]*(1.0 - psi) );
				rowProd[iRow] *= psi;
			}
		}
	}
}
void NumerUtil::phi2p(const MatrixView &Phi, MatrixView &P) const{
#ifndef PKG_DEBUG_OFF
	if ( ( Phi.getNcols()+1 != P.getNcols() ) || ( Phi.getNrows() != P.getNrows() ) ){
		throw string("ERROR: Phi (") + to_string( Phi.getNrows() ) + "x" + to_string( Phi.getNcols() ) +  string(") and P (") + to_string( P.getNrows() ) + "x" + to_string( P.getNcols() ) + string(") dimensions incompatible in phi2p(MatrixView &, MatrixView &)");
	}
#endif
	for (size_t m = 0; m < Phi.getNcols(); m++) {
		for (size_t iRow = 0; iRow < Phi.getNrows(); iRow++) {
			P.setElem( iRow, m, logistic( Phi.getElem(iRow, m) ) );
		}
	}
	// Re-weight the P using the Betancourt (2012) algorithm
	vector<double> rowProd(Phi.getNrows(), 0.0);
	for (size_t m = 0; m < P.getNcols(); m++) {
		for (size_t iRow = 0; iRow < P.getNrows(); iRow++) {
			if (m == 0){
				rowProd[iRow] = P.getElem(iRow, m);
				P.setElem(iRow, m, 1.0 - rowProd[iRow]);
			} else if ( m == Phi.getNcols() ){
				P.setElem(iRow, m, rowProd[iRow]);
			} else {
				double psi = P.getElem(iRow, m);
				P.setElem( iRow, m, rowProd[iRow]*(1.0 - psi) );
				rowProd[iRow] *= psi;
			}
		}
	}
}
void NumerUtil::w2p(const MatrixView &W, MatrixView &P) const{
#ifndef PKG_DEBUG_OFF
	if ( ( W.getNcols()+1 != P.getNcols() ) || ( W.getNrows() != P.getNrows() ) ){
		throw string("ERROR: W (") + to_string( W.getNrows() ) + "x" + to_string( W.getNcols() ) +  string(") and P (") + to_string( P.getNrows() ) + "x" + to_string( P.getNcols() ) + string(") dimensions incompatible in w2p(MatrixView &, MatrixView &)");
	}
#endif
	// Re-weight the P using the Betancourt (2012) algorithm
	vector<double> rowProd(W.getNrows(), 0.0);
	for (size_t m = 0; m < P.getNcols(); m++) {
		for (size_t iRow = 0; iRow < P.getNrows(); iRow++) {
			if (m == 0){
				rowProd[iRow] = W.getElem(iRow, m);
				P.setElem(iRow, m, 1.0 - rowProd[iRow]);
			} else if ( m == W.getNcols() ){
				P.setElem(iRow, m, rowProd[iRow]);
			} else {
				double psi = W.getElem(iRow, m);
				P.setElem( iRow, m, rowProd[iRow]*(1.0 - psi) );
				rowProd[iRow] *= psi;
			}
		}
	}
}
double NumerUtil::dotProd(const vector<double> &v) const{
	double dotProd = 0.0;
	for (auto &element : v) {
		dotProd += element*element;
	}
	return dotProd;
}
double NumerUtil::dotProd(const vector<double> &v1, const vector<double> &v2) const{
	double dotProd = 0.0;
	auto v1It = v1.begin();
	auto v2It = v2.begin();
	// this ensures that we don't go beyond one of the vectors; worth declaring the iterators outside the loop and the extra operations
	for ( ; (v1It != v1.end()) && (v2It != v2.end()); ++v1It, ++v2It) {
		dotProd += (*v1It)*(*v2It);
	}
	return dotProd;
}
 void NumerUtil::updateWeightedMean(const double &xn, const double &wn, double &mu, double &w) const{
	const double a = mu*w;
	w += wn;
	mu = (a + wn*xn)/w;
}
