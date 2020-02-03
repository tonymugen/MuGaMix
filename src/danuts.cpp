/*
 * Copyright (c) 2018 Anthony J. Greenberg
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

/// NUTS with dual averaging
/** \file
 * \author Anthony J. Greenberg
 * \copyright Copyright (c) 2018 Anthony J. Greenberg
 * \version 1.0
 *
 * Class implementation for the No-U-Turn Sampler with dual averaging.
 *
 */

#include <vector>
#include <string>
#include <limits>
#include <cmath>

#include "danuts.hpp"
#include "random.hpp"

using namespace BayesicSpace;

using std::vector;
using std::string;
using std::numeric_limits;
using std::fpclassify;
using std::signbit;

// static members
const double SamplerNUTS::deltaMax_ = 1000.0;
const double SamplerNUTS::delta_    = 0.6;
const double SamplerNUTS::t0_       = 10.0;
const double SamplerNUTS::gamma_    = 0.05;
const double SamplerNUTS::negKappa_ = -0.75;
const uint64_t SamplerNUTS::mask_   = static_cast<uint64_t>(0x01);

/** \brief Vector self-dot-product
 *
 * \param[in] v vector
 * \return dot-product value
 */
double dotProd(const vector<double> &v){
	double dotProd = 0.0;
	for (auto &element : v) {
		dotProd += element*element;
	}
	return dotProd;
}
/** \brief Dot-product of two vectors
 *
 * \param[in] v1 vector 1
 * \param[in] v2 vector 2
 * \return dot-product value
 */
double dotProd(const vector<double> &v1, const vector<double> &v2){
	double dotProd = 0.0;
	auto v1It = v1.begin();
	auto v2It = v2.begin();
	// this ensures that we don't go beyond one of the vectors; worth declaring the iterators outside the loop and the extra operations
	for ( ; (v1It != v1.end()) && (v2It != v2.end()); ++v1It, ++v2It) {
		dotProd += (*v1It)*(*v2It);
	}
	return dotProd;
}
/** \brief Weighted mean update function
 *
 * Takes the current weighted mean and updates using the new data point and weight. The formula is
 *
 * \f$
 *     \bar{\mu}_n = \dfrac{\bar{\mu}_{n-1}\sum_{i=1}^{n-1}w_i + w_n x_n}{\sum_{i=1}^{n-1}w_i + w_n}
 * \f$
 *
 * \param[in] xn new point \f$ x_n \f$
 * \param[in] wn weight \f$ w_n \f$
 * \param[out] mu new mean
 * \param[out] w new weight
 *
 */
 void updateWeightedMean(const double &xn, const double &wn, double &mu, double &w){
	const double a = mu*w;

	w += wn;
	mu = (a + wn*xn)/w;
}

SamplerNUTS::SamplerNUTS(SamplerNUTS &&in) : epsilon_{in.epsilon_}, mu_{in.mu_}, nH0_{in.nH0_}, m_{in.m_}, Hprevious_{in.Hprevious_}, logEpsBarPrevious_{in.logEpsBarPrevious_}, epsWMN_{in.epsWMN_}, currW_{in.currW_}, firstAdapt_{in.firstAdapt_}, firstUpdate_{in.firstUpdate_}, model_{in.model_}, theta_{in.theta_} {
	in.model_ = nullptr;
	in.theta_ = nullptr;
}
SamplerNUTS& SamplerNUTS::operator=(SamplerNUTS &&in){
	if (&in == this) {
		return *this;
	}
	epsilon_           = in.epsilon_;
	mu_                = in.mu_;
	nH0_               = in.nH0_;
	m_                 = in.m_;
	Hprevious_         = in.Hprevious_;
	logEpsBarPrevious_ = in.logEpsBarPrevious_;
	epsWMN_            = in.epsWMN_;
	currW_             = in.currW_;
	firstAdapt_        = in.firstAdapt_;
	firstUpdate_       = in.firstUpdate_;
	model_             = in.model_;
	theta_             = in.theta_;
	in.model_ = nullptr;
	in.theta_ = nullptr;

	return *this;
}

void SamplerNUTS::findInitialEpsilon_(){
	// Algorithm 4 in Hoffman and Gelman
	// epsilon_ initialized in the constructor initialization list
	vector<double> r0;
	for (size_t i = 0; i < theta_->size(); i++) {
		r0.push_back(rng_.rnorm());
	}

	vector<double> thetaPrime(*theta_);
	vector<double> rPrime(r0);
	leapfrog_(thetaPrime, rPrime, epsilon_);
	char a = '\0'; // instead of a boolean to guarantee one byte

	// have to make sure no weirdness (like NaN or Inf) comes out of the log-posterior evaluation
	double logp      = model_->logPost(*theta_);
	double logpPrime = model_->logPost(thetaPrime);
	int fpClsLP      = fpclassify(logp);
	int fpClsLPP     = fpclassify(logpPrime);
	if (fpClsLPP == FP_NAN) {
		throw string("log-posterior for thetaPrime evaluates to NaN in findInitialEpsilon_");
	} else if (fpClsLP == FP_NAN) {
		throw string("log-posterior for theta evaluates to NaN in findInitialEpsilon_");
	} else if (fpClsLPP == FP_INFINITE) {
		if (signbit(logpPrime)) { // logpPrime is -Inf
			if ( (fpClsLP == FP_INFINITE) && (signbit(logp)) ) { // only if logp == -Inf a = 1
				a = '1';
			}
		} else { // logpPrime is +Inf
			throw string("log-posterior evaluates to +Inf in findInitialEpsilon_. This should never happen. Check your implementation.");
		}
	} else if (fpClsLP == FP_INFINITE) {
		if (signbit(logp)) { // logp is -Inf; the only condition (| logpPrime not +/- Inf as tested above) when logpPrime - logp > -1.69...
			a = '1';
		}
	} else {
		logp      -= 0.5*dotProd(r0);
		logpPrime -= 0.5*dotProd(rPrime);
		a          = ((logpPrime - logp) > -0.6931472 ? '1' : '\0' );  // -0.6931472 = log(0.5); taking a log of the I() condition; '\0' equivalent to a = -1.0 in Algorithm 4
	}

	if (a) { // a = 1.0
		for (uint16_t i = 0; i < 7; i++) { // do not do more than seven doublings; initial values may be wrong and result in epsilon_ too large for regular operation
			epsilon_   = 2.0*epsilon_;
			thetaPrime = *theta_;
			rPrime     = r0;
			leapfrog_(thetaPrime, rPrime, epsilon_);
			logpPrime = model_->logPost(thetaPrime);
			fpClsLPP  = fpclassify(logpPrime);
			if (fpClsLPP == FP_NAN) {
				throw string("log-posterior for thetaPrime evaluates to NaN in findInitialEpsilon_");
			} else if (fpClsLPP == FP_INFINITE) {
				if (signbit(logpPrime)) { // logpPrime is -Inf; definitely break out
					break;
				} else { // logpPrime is +Inf
					throw string("log-posterior evaluates to +Inf in findInitialEpsilon_. This should never happen. Check your implementation.");
				}
			} else {
				logpPrime -= 0.5*dotProd(rPrime);
				if ((logpPrime - logp) > -0.6931472) {  // take a log of the while() test inequality; a = 1.0 so the direction is the same as in the description
					break;
				}
			}
		}
	} else { // a = -1.0
		for (uint16_t i = 0; i < 7; i++) { // do not do more than seven halves or epsilon_ will be too small
			epsilon_   = 0.5*epsilon_;
			thetaPrime = *theta_;
			rPrime     = r0;
			leapfrog_(thetaPrime, rPrime, epsilon_);
			logpPrime = model_->logPost(thetaPrime);
			fpClsLPP  = fpclassify(logpPrime);
			if (fpClsLPP == FP_NAN) {
				throw string("log-posterior for thetaPrime evaluates to NaN in findInitialEpsilon_");
			} else if (fpClsLPP == FP_INFINITE) {
				if (signbit(logpPrime)) { // logpPrime is -Inf; make epsilon_ smaller to see if I don't shoot that low
					continue;
				} else { // logpPrime is +Inf
					throw string("log-posterior evaluates to +Inf in findInitialEpsilon_. This should never happen. Check your implementation.");
				}
			} else {
				logpPrime -= 0.5*dotProd(rPrime);
				if ((logpPrime - logp) < -0.6931472) {  // take a log of the while() test inequality; a = -1.0, so the inequality is switched
					break;
				}
			}
		}
	}
	mu_ = 2.302585 + log(epsilon_);  // log(10epsilon_0)
}

void SamplerNUTS::leapfrog_(vector<double> &theta, vector<double> &r, const double &epsilon){
	vector<double> thtGrad;  // Make sure that the model implementing the gradient resizes it properly!
	model_->gradient(theta, thtGrad);

	for (size_t j = 0; j < theta.size(); j++) {
		r[j]     += 0.5*epsilon*thtGrad[j];  // half-step update of r
		theta[j] += epsilon*r[j];            // leapfrog update of theta
	}
	model_->gradient(theta, thtGrad);
	// one more half-step update of r
	for (size_t k = 0; k < theta.size(); k++) {
		r[k] += 0.5*epsilon*thtGrad[k];
	}

}

void SamplerNUTS::buildTreePos_(const vector<double> &theta, const vector<double> &r, const double &lu, const double &epsilon, const uint16_t &j, vector<double> &thetaPlus, vector<double> &rPlus, const vector<double> &thetaMinus, const vector<double> &rMinus, vector<double> &thetaPrime, double &nPrime, char &s){
	if (j == 0) {
		thetaPrime = theta;
		vector<double> rPrime(r);
		leapfrog_(thetaPrime, rPrime, epsilon);

		thetaPlus      = thetaPrime;
		rPlus          = move(rPrime);
		double testVal = model_->logPost(thetaPrime);
		int fpClsTV    = fpclassify(testVal);
		if (fpClsTV == FP_NAN) {
			throw string("Log-posterior evaluates to NaN in buildTreePos_");
		} else if (fpClsTV == FP_INFINITE) {
			if (signbit(testVal)) { // testVal is -Inf, definitely smaller that lu
				nPrime = 0.0;
				s      = '\0';
			} else {
				throw string("log-posterior evaluates to +Inf in buildTreePos_. This should never happen. Check your implementation.");
			}
		} else {
			testVal -= 0.5*dotProd(rPrime);
			nPrime   = (lu <= testVal ? 1.0 : 0.0);
			s        = (lu < (deltaMax_ + testVal) ? '1' : '\0');
		}
	} else {
		// recursion
		buildTreePos_(theta, r, lu, epsilon, j-1, thetaPlus, rPlus, thetaMinus, rMinus, thetaPrime, nPrime, s);
		if (s) {
			char sDPrm   = '1';        // s''
			double nDprm = 0.0;        // n''
			vector<double> thetaDprm;  // theta''
			buildTreePos_(thetaPlus, rPlus, lu, epsilon, j-1, thetaPlus, rPlus, thetaMinus, rMinus, thetaDprm, nDprm, sDPrm);
			nPrime += nDprm;
			if ( nPrime && (rng_.runif() <= nDprm/nPrime) ) { // nPrime now nPrime+nDprm
				thetaPrime = move(thetaDprm);
			}
			if (sDPrm) { // only now necessary to test the dot-product condition; equivalent to s''I(...) in Algorithm 3
				vector<double> thetaDiff;
				// theta^+ - theta^-
				for (size_t jTht = 0; jTht < thetaPlus.size(); jTht++) {
					thetaDiff.push_back(thetaPlus[jTht] - thetaMinus[jTht]);
				}
				double dot = dotProd(thetaDiff, rMinus);
				if (dot >= 0.0) { // only then it is necessary to do the second dot product
					dot = dotProd(thetaDiff, rPlus);
					if (dot < 0.0) {
						s = '\0';
					}
				} else {
					s = '\0';
				}
			} else {
				s = '\0';
			}
		}
	}

}

void SamplerNUTS::buildTreeNeg_(const vector<double> &theta, const vector<double> &r, const double &lu, const double &epsilon, const uint16_t &j, const vector<double> &thetaPlus, const vector<double> &rPlus, vector<double> &thetaMinus, vector<double> &rMinus, vector<double> &thetaPrime, double &nPrime, char &s){
	if (j == 0) {
		thetaPrime = theta;
		vector<double> rPrime(r);
		leapfrog_(thetaPrime, rPrime, epsilon);

		thetaMinus     = thetaPrime;
		rMinus         = move(rPrime);
		double testVal = model_->logPost(thetaPrime);
		int fpClsTV    = fpclassify(testVal);
		if (fpClsTV == FP_NAN) {
			throw string("log-posterior evaluates to NaN in buildTreeNeg_");
		} else if (fpClsTV == FP_INFINITE) {
			if (signbit(testVal)) { // testVal is -Inf
				nPrime = 0.0;
				s      = '\0';
			} else { // testVal is +Inf
				throw string("log-posterior evaluates to +Inf in buildTreeNeg_. This should never happen. Check your implementation.");
			}
		} else {
			testVal -= 0.5*dotProd(rPrime);
			nPrime   = (lu <= testVal ? 1.0 : 0.0);
			s        = (lu < (deltaMax_ + testVal) ? '1' : '\0');
		}
	} else {
		// recursion
		buildTreeNeg_(theta, r, lu, epsilon, j-1, thetaPlus, rPlus, thetaMinus, rMinus, thetaPrime, nPrime, s);
		if (s) {
			char sDPrm   = '1';        // s''
			double nDprm = 0.0;        // n''
			vector<double> thetaDprm;  // theta''
			buildTreeNeg_(thetaMinus, rMinus, lu, epsilon, j-1, thetaPlus, rPlus, thetaMinus, rMinus, thetaDprm, nDprm, sDPrm);
			nPrime += nDprm;
			if ( nPrime && (rng_.runif() <= nDprm/nPrime) ) { // nPrime now nPrime+nDprm
				thetaPrime = move(thetaDprm);
			}
			if (sDPrm) { // only now necessary to test the dot-product condition; equivalent to s''I(...) in Algorithm 3
				vector<double> thetaDiff;
				// theta^+ - theta^-
				for (size_t jTht = 0; jTht < thetaPlus.size(); jTht++) {
					thetaDiff.push_back(thetaPlus[jTht] - thetaMinus[jTht]);
				}
				double dot = dotProd(thetaDiff, rMinus);
				if (dot >= 0.0) { // only then it is necessary to do the second dot product
					dot = dotProd(thetaDiff, rPlus);
					if (dot < 0.0) {
						s = '\0';
					}
				} else {
					s = '\0';
				}
			} else {
				s = '\0';
			}
		}
	}

}

void SamplerNUTS::buildTreePos_(const vector<double> &theta, const vector<double> &r, const double &lu, const double &epsilon, const uint16_t &j, vector<double> &thetaPlus, vector<double> &rPlus, const vector<double> &thetaMinus, const vector<double> &rMinus, vector<double> &thetaPrime, double &nPrime, char &s, double &alphaPrime, double &nAlphaPrime){
	if (j == 0) {
		thetaPrime = theta;
		vector<double> rPrime(r);
		leapfrog_(thetaPrime, rPrime, epsilon);
		thetaPlus      = thetaPrime;
		rPlus          = move(rPrime);
		double testVal = model_->logPost(thetaPrime);
		int fpClsTV    = fpclassify(testVal);
		if (fpClsTV == FP_NAN) {
			throw string("log-posterior evluates to NaN in adaptive buildTreePos_");
		} else if (fpClsTV == FP_INFINITE) {
			if (signbit(testVal)) { // log-post is -Inf
				nPrime     = 0.0;
				s          = '\0';
				alphaPrime = 0.0; // exp(-Inf)
			} else { // log-post is +Inf
				throw string("log-posterior evaluates to +Inf in adaptive buildTreePos_. This should never happen. Check your implementation.");
			}
		} else {
			testVal -= 0.5*dotProd(rPrime);
			nPrime     = (lu <= testVal ? 1.0 : 0.0);
			s          = (lu < (deltaMax_ + testVal) ? '1' : '\0');
			const double pDiff = testVal - nH0_;
			alphaPrime = (pDiff >= 0.0 ? 1.0 : exp(pDiff));
		}
		nAlphaPrime  = 1.0;

	} else {
		// recursion
		buildTreePos_(theta, r, lu, epsilon, j-1, thetaPlus, rPlus, thetaMinus, rMinus, thetaPrime, nPrime, s, alphaPrime, nAlphaPrime);
		if (s) {
			char sDPrm        = '\0';  // s''
			double nDprm      = 0.0;   // n''
			double alphaDprm  = 0.0;   // alpha''
			double nAlphaDprm = 0.0;   // n_alpha''
			vector<double> thetaDprm;  // theta''
			buildTreePos_(thetaPlus, rPlus, lu, epsilon, j-1, thetaPlus, rPlus, thetaMinus, rMinus, thetaDprm, nDprm, sDPrm, alphaDprm, nAlphaDprm);
			alphaPrime  += alphaDprm;
			nAlphaPrime += nAlphaDprm;
			nPrime      += nDprm;
			if ( (nPrime > 0.0) && (nDprm > 0.0) && (rng_.runif() <= nDprm/nPrime)) { // nPrime now nPrime+nDprm
				thetaPrime = move(thetaDprm);
			}
			if (sDPrm) { // only now necessary to test the dot-product condition; equivalent to s''I(...) in Algorithm 3 and 6
				vector<double> thetaDiff;
				// theta^+ - theta^-
				for (size_t jTht = 0; jTht < thetaPlus.size(); jTht++) {
					thetaDiff.push_back(thetaPlus[jTht] - thetaMinus[jTht]);
				}
				double dot = dotProd(thetaDiff, rMinus);
				if (dot >= 0.0) { // only then it is necessary to do the second dot product
					dot = dotProd(thetaDiff, rPlus);
					if (dot < 0.0) {
						s = '\0';
					}
				} else {
					s = '\0';
				}
			} else {
				s = '\0';
			}
		}
	}

}
void SamplerNUTS::buildTreeNeg_(const vector<double> &theta, const vector<double> &r, const double &lu, const double &epsilon, const uint16_t &j, const vector<double> &thetaPlus, const vector<double> &rPlus, vector<double> &thetaMinus, vector<double> &rMinus, vector<double> &thetaPrime, double &nPrime, char &s, double &alphaPrime, double &nAlphaPrime){
	if (j == 0) {
		thetaPrime = theta;
		vector<double> rPrime(r);
		leapfrog_(thetaPrime, rPrime, epsilon);
		thetaMinus     = thetaPrime;
		rMinus         = move(rPrime);
		double testVal = model_->logPost(thetaPrime);
		int fpClsTV    = fpclassify(testVal);
		if (fpClsTV == FP_NAN) {
			throw string("log-posterior evluates to NaN in adaptive buildTreeNeg_");
		} else if (fpClsTV == FP_INFINITE) {
			if (signbit(testVal)) { // log-post is -Inf
				nPrime     = 0.0;
				s          = '\0';
				alphaPrime = 0.0; // exp(-Inf)
			} else { // log-post is +Inf
				throw string("log-posterior evaluates to +Inf in adaptive buildTreeNeg_. This should never happen. Check your implementation.");
			}
		} else {
			testVal -= 0.5*dotProd(rPrime);
			nPrime     = (lu <= testVal ? 1.0 : 0.0);
			s          = (lu < (deltaMax_ + testVal) ? '1' : '\0');
			const double pDiff = testVal - nH0_;
			alphaPrime = (pDiff >= 0.0 ? 1.0 : exp(pDiff));
		}
		nAlphaPrime  = 1.0;

	} else {
		// recursion
		buildTreeNeg_(theta, r, lu, epsilon, j-1, thetaPlus, rPlus, thetaMinus, rMinus, thetaPrime, nPrime, s, alphaPrime, nAlphaPrime);
		if (s) {
			char sDPrm        = '\0';  // s''
			double nDprm      = 0.0;   // n''
			double alphaDprm  = 0.0;   // alpha''
			double nAlphaDprm = 0.0;   // n_alpha''
			vector<double> thetaDprm;  // theta''
			buildTreeNeg_(thetaMinus, rMinus, lu, epsilon, j-1, thetaPlus, rPlus, thetaMinus, rMinus, thetaDprm, nDprm, sDPrm, alphaDprm, nAlphaDprm);
			alphaPrime  += alphaDprm;
			nAlphaPrime += nAlphaDprm;
			nPrime      += nDprm;
			if ( (nPrime > 0.0) && (nDprm > 0.0) && (rng_.runif() <= nDprm/nPrime)) { // nPrime now nPrime+nDprm
				thetaPrime = move(thetaDprm);
			}
			if (sDPrm) { // only now necessary to test the dot-product condition; equivalent to s''I(...) in Algorithm 3 and 6
				vector<double> thetaDiff;
				// theta^+ - theta^-
				for (size_t jTht = 0; jTht < thetaPlus.size(); jTht++) {
					thetaDiff.push_back(thetaPlus[jTht] - thetaMinus[jTht]);
				}
				double dot = dotProd(thetaDiff, rMinus);
				if (dot >= 0.0) { // only then it is necessary to do the second dot product
					dot = dotProd(thetaDiff, rPlus);
					if (dot < 0.0) {
						s = '\0';
					}
				} else {
					s = '\0';
				}
			} else {
				s = '\0';
			}
		}
	}

}

uint32_t SamplerNUTS::adapt(){
	// Putting this here despite the small overhead of testing every time I do an adaptation step, so that the initialization is hidden from the user and therefore foolproof
	if (firstAdapt_	) {
		findInitialEpsilon_();
		firstAdapt_ = false;
	}
	// following the notation in Hoffman and Gelman
	// epsilon_0 and mu already set in the constructor (calling FindInitialEpsilon)
	char s        = '1'; // the stopping condition; since sizeof(bool) is implementation-defined, I opt for using one byte for sure
	uint16_t j    = 0;   // tree depth
	double n      = 1.0; // n; using a double straight away so that I do not have to re-cast for division
	double alpha  = 0.0;
	double nAlpha = 0.0;
	// sampling r_0
	vector<double> r0;
	for (size_t i = 0; i < theta_->size(); i++) {
		r0.push_back(rng_.rnorm());
	}
	vector<double> rPlus(r0);
	vector<double> rMinus(r0);
	// sampling the log-slice variable
	nH0_        = model_->logPost(*theta_);
	int fpClsH0 = fpclassify(nH0_);
	// check sanity of log-posterior evaluation
	if (fpClsH0 == FP_NAN) {
		throw string("log-posterior evlauates to NaN in the adaptation phase");
	} else if (fpClsH0 == FP_INFINITE) {

		if (signbit(nH0_)) { // logpost is -Inf
			// try to find theta values that give a finite log-posterior. Stop when this happens, but give up after 100 attempts and hope the next round will be better. Return 0 as tree depth.
			for (uint16_t i = 0; i < 100; i++) {
				leapfrog_(*theta_, r0, epsilon_);
				if (fpclassify(model_->logPost(*theta_)) == FP_NORMAL) {
					break;
				}
			}
			const double mt0    = m_ + t0_;
			Hprevious_          = (1.0 - 1.0/mt0)*Hprevious_ + delta_/mt0;
			const double logEps = mu_ - (sqrt(m_)*Hprevious_)/gamma_;
			epsilon_            = exp(logEps);
			const double mPwr   = pow(m_, negKappa_);
			logEpsBarPrevious_  = mPwr*logEps + (1.0 - mPwr)*logEpsBarPrevious_;
			updateWeightedMean(epsilon_, sqrt(m_), epsWMN_, currW_);
			m_ += 1.0;
			return 0;
		} else { // logpost is +Inf, which is bad
			throw ("log-posterior evaluates to +Inf in adapt(), which should never happen. Check your posterior function implementation.");
		}

	}
	nH0_ -= 0.5*dotProd(r0);
	const double lu = log(rng_.runifnz()) + nH0_;   // log(slice variable)

	vector<double> thetaPlus(*theta_);
	vector<double> thetaMinus(*theta_);
	vector<double> thetaPrime;
	// theta_ will be theta^{m-1}
	while (s) {
		double nPrime = 0.0;
		char sPrime   = '\0';
		if (rng_.ranInt()&static_cast<uint64_t>(0x01)) { // testing if the last bit is set; should be a 50/50 chance, so in effect sampling U{-1,1}
			// positive step; copy negative variables to prevent modification
			buildTreePos_(thetaPlus, rPlus, lu, epsilon_, j, thetaPlus, rPlus, thetaMinus, rMinus, thetaPrime, nPrime, sPrime, alpha, nAlpha);
		} else {
			// negative step; copy positive variables to prevent modification
			buildTreeNeg_(thetaMinus, rMinus, lu, -epsilon_, j, thetaPlus, rPlus, thetaMinus, rMinus, thetaPrime, nPrime, sPrime, alpha, nAlpha);
		}
		if (sPrime) {
			if ( (nPrime >= n) || (rng_.runif() <= nPrime/n) ) {
				(*theta_) = move(thetaPrime);
			}
			vector<double> thetaDiff;
			// theta^+ - theta^-
			for (size_t jTht = 0; jTht < thetaPlus.size(); jTht++) {
				thetaDiff.push_back(thetaPlus[jTht] - thetaMinus[jTht]);
			}
			double dot = dotProd(thetaDiff, rMinus);
			if (dot >= 0.0) { // only then it is necessary to do the second dot product
				dot = dotProd(thetaDiff, rPlus);
				if (dot < 0.0) {
					break; // s == 0
				}
			} else {
				break; // s == 0
			}
		} else {
			break; // s == 0
		}
		n += nPrime;

		if (j >= 8) { // too many doublings; nudge the epsilon to a larger value
			alpha  = 1.0;
			nAlpha = 1.0;
			break;
		}
		j++;
	}

	const double mt0    = m_ + t0_;
	Hprevious_          = (1.0 - 1.0/mt0)*Hprevious_ + (delta_ - alpha/nAlpha)/mt0; // buildTree_ execution guarantees that nAlpha >= 1.0
	const double logEps = mu_ - (sqrt(m_)*Hprevious_)/gamma_;
	epsilon_            = exp(logEps);
	const double mPwr   = pow(m_, negKappa_);
	logEpsBarPrevious_  = mPwr*logEps + (1.0 - mPwr)*logEpsBarPrevious_;
	updateWeightedMean(epsilon_, sqrt(m_), epsWMN_, currW_);
	m_ += 1.0;

	return j;
}

uint32_t SamplerNUTS::update() {
	// I am adding this step here despite the small overhead to make the flow hidden from the user as much as possible, eliminating potential errors
	if (firstUpdate_) {
		if (firstAdapt_) {
			firstUpdate_ = false;   // in case there were no adapt runs, leave epsilon_ as the constructor-assigned value
		} else {
			epsilon_     = epsWMN_;
			firstUpdate_ = false;
		}
	}

	// following the notation in Hoffman and Gelman
	char s     = '1'; // the stopping condition; since sizeof(bool) is implementation-defined, I opt for using one byte for sure
	uint16_t j = 0;   // tree depth
	double n   = 1.0; // n; using a double straight away so that I do not have to re-cast for division
	// sampling r_0
	vector<double> rPlus;
	for (size_t i = 0; i < theta_->size(); i++) {
		rPlus.push_back(rng_.rnorm());
	}
	vector<double> rMinus(rPlus);
	// sampling the log-slice variable
	const double lPost = model_->logPost(*theta_);
	int fpClsLP = fpclassify(lPost);
	// check sanity of log-posterior evaluation
	if (fpClsLP == FP_NAN) {
		throw string("log-posterior evlauates to NaN in the update phase");
	} else if (fpClsLP == FP_INFINITE) {

		if (signbit(nH0_)) { // logpost is -Inf
			// try to find theta values that give a finite log-posterior. Stop when this happens, but give up after 100 attempts and hope the next round will be better. Return 0 as tree depth.
			for (uint16_t i = 0; i < 100; i++) {
				leapfrog_(*theta_, rPlus, epsilon_);
				if (fpclassify(model_->logPost(*theta_)) == FP_NORMAL) {
					break;
				}
			}
			return 0;
		} else { // logpost is +Inf, which is bad
			throw ("log-posterior evaluates to +Inf in update(), which should never happen. Check your posterior function implementation.");
		}

	}
	const double lu = log(rng_.runifnz()) + lPost - 0.5*dotProd(rPlus);   // log(slice variable)

	vector<double> thetaPlus(*theta_);
	vector<double> thetaMinus(*theta_);
	vector<double> thetaPrime;
	// theta_ will play the role of theta^m in Algorithm 3
	while (s) {
		double nPrime = 0.0;
		char sPrime   = '\0';
		if (rng_.ranInt()&mask_) { // testing if the last bit is set; should be a 50/50 chance, so in effect sampling U{-1,1}
			// positive step
			buildTreePos_(thetaPlus, rPlus, lu, epsilon_, j, thetaPlus, rPlus, thetaMinus, rMinus, thetaPrime, nPrime, sPrime);
		} else {
			// negative step
			buildTreeNeg_(thetaMinus, rMinus, lu, -epsilon_, j, thetaPlus, rPlus, thetaMinus, rMinus, thetaPrime, nPrime, sPrime);
		}
		if (sPrime) {
			if ( (nPrime >= n) || (rng_.runif() <= nPrime/n) ) {
				(*theta_) = move(thetaPrime);
			}
			vector<double> thetaDiff;
			// theta^+ - theta^-
			for (size_t jTht = 0; jTht < thetaPlus.size(); jTht++) {
				thetaDiff.push_back(thetaPlus[jTht] - thetaMinus[jTht]);
			}
			double dot = dotProd(thetaDiff, rMinus);
			if (dot >= 0.0) { // only then it is necessary to do the second dot product
				dot = dotProd(thetaDiff, rPlus);
				if (dot < 0.0) {
					break; // s == 0
				}
			} else {
				break; // s == 0
			}
		} else {
			break; // s == 0
		}
		n += nPrime;
		j++;
	}
	return j;

}

// SamplerMetro methods
SamplerMetro::SamplerMetro(SamplerMetro &&in) :  model_{in.model_}, theta_{in.theta_} {
	in.model_ = nullptr;
	in.theta_ = nullptr;
}
SamplerMetro& SamplerMetro::operator=(SamplerMetro &&in){
	if (&in == this) {
		return *this;
	}
	model_    = in.model_;
	theta_    = in.theta_;
	in.model_ = nullptr;
	in.theta_ = nullptr;

	return *this;
}

uint32_t SamplerMetro::adapt(){
	vector<double> thetaPrime = *theta_;
	for (auto &t : thetaPrime) {
		t += rng_.rnorm();
	}
	double lAlpha = model_->logPost(thetaPrime) - model_->logPost(*theta_);
	double lU     = log(rng_.runifnz());
	if (lU < lAlpha) {
		(*theta_) = move(thetaPrime);
		return 1;
	} else {
		return 0;
	}
}

uint32_t SamplerMetro::update(){
	vector<double> thetaPrime = *theta_;
	for (auto &t : thetaPrime) {
		t += 0.01*rng_.rnorm();
	}
	double lAlpha = model_->logPost(thetaPrime) - model_->logPost(*theta_);
	double lU     = log(rng_.runifnz());
	if (lU < lAlpha) {
		(*theta_) = move(thetaPrime);
		return 1;
	} else {
		return 0;
	}
}

