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
 * Class definition and interface documentation to generate Markov chains for inference from multitrait Gaussian mixture models. Dual-averaging NUTS and Metropolis samplers for parameters groups are included within a Gibbs sampler.
 *
 */
#ifndef lme_hpp
#define lme_hpp

#include <vector>
#include <cmath>

#include "index.hpp"
#include "random.hpp"
#include "model.hpp"
#include "sampler.hpp"
#include "danuts.hpp"
#include "matrixView.hpp"

using std::vector;

namespace BayesicSpace {
	// forward declarations
	class MumiLoc;
	class MumiISig;

	class WrapMMM;

	/** \brief Mixture model for location parameters
	 *
	 * Implements log-posterior and gradient for the location parameters of the mixture model.
	 *
	 */
	class MumiLoc : public Model {
	public:
		/** \brief Default constructor */
		MumiLoc() : Model(), hierInd_{nullptr}, tau0_{0.0}, tauP_{0.0} {};
		/** \brief Constructor
		 *
		 * \parameter[in] yVec pointer vectorized data matrix
		 * \parameter[in] iSigVec pointer to vectorized inverse-covariance matrix collection
		 * \parameter[in] d number of traits
		 * \parameter[in] hierInd pointer to vector of hierarchical indexes
		 * \parameter[in] xVec pointer to vectorized covariate predictor matrix
		 * \parameter[in] tau fixed prior for the unmodeled ("fixed") effects and population means
		 */
		MumiLoc(const vector<double> *yVec, const vector<double> *iSigVec, const size_t &d, const vector<Index> *hierInd, const vector<double> *xVec, const double &tau);
		/** \brief Destructor */
		~MumiLoc(){};

		/** \brief Copy constructor (deleted) */
		MumiLoc(const MumiLoc &in) = delete;
		/** \brief Copy assignment (deleted) */
		MumiLoc& operator=(const MumiLoc &in) = delete;
		/** \brief Move constructor
		 *
		 * \param[in] in object to move
		 */
		MumiLoc(MumiLoc &&in) : Y_{move(in.Y_)}, ISigE_{move(in.ISigE_)}, ISigA_{move(in.ISigA_)}, X_{move(in.X_)}, hierInd_{in.hierInd_}, tau0_{in.tau0_}, tauP_{in.tauP_} {in.hierInd_ = nullptr;};
		/** \brief Move assignment operator
		 *
		 * \param[in] in object to be moved
		 * \return target object
		 */
		MumiLoc& operator=(MumiLoc &&in);
		/** \brief Log-posterior function
		 *
		 * Returns the value of the log-posterior given the data provided at construction and the passed-in parameter vector. The parameter vector has the covariates, line means, and population means in that order.
		 *
		 * \param[in] theta parameter vector
		 * \return Value of the log-posterior
		 */
		double logPost(const vector<double> &theta) const;
		/** \brief Gradient of the log-posterior
		 *
		 * Calculates the patial derivative of the log-posterior for each element in the provided parameter vector.
		 *
		 * \param[in] theta parameter vector
		 * \param[out] grad partial derivative (gradient) vector
		 *
		 */
		void gradient(const vector<double> &theta, vector<double> &grad) const;

	protected:
		/** \brief Matrix view of data */
		MatrixViewConst Y_;
		/** \brief Matrix view of the error inverse-covariance */
		MatrixViewConst ISigE_;
		/** \brief Matrix view of the line inverse-covariance */
		MatrixViewConst ISigA_;
		/** \brief Matrix view of covariate predictors */
		MatrixViewConst X_;
		/** \brief Pointer to vector of indexes connecting hierarchy levels */
		const vector<Index> *hierInd_;
		/** \brief Fixed prior precision for unmodeled effects */
		const double tau0_;
		/** \brief Fixed prior precision for population means */
		const double tauP_;
	};

	/** \brief Model for inverse covariances
	 *
	 * Implements log-posterior and gradient for inverse covariances.
	 *
	 */
	class MumiISig : public Model {
	public:
		/** \brief Default constructor */
		MumiISig(): Model(), vY_{nullptr}, vTheta_{nullptr}, vX_{nullptr}, hierInd_{nullptr} {};
		/** \brief Constructor
		 *
		 * \parameter[in] y pointer to data
		 * \parameter[in] theta pointer to vector of location parameters
		 * \parameter[in] y2loc pointer to vector of factors connecting data to location parameters
		 * \parameter[in] loc2prior pointer to `Index` connecting location parameters to their priors
		 *
		 */
		MumiISig(const vector<double> *yVec, const vector<double> *vTheta, const vector<double> *xVec, const vector<Index> *hierInd) : Model(), vY_{yVec}, vTheta_{vTheta}, vX_{xVec}, hierInd_{hierInd} {};
		/** \brief Log-posterior function
		 *
		 * Returns the value of the log-posterior given the data provided at construction and the passed-in parameter vector.
		 *
		 * \param[in] viSig parameter vector
		 * \return Value of the log-posterior
		 */
		virtual double logPost(const vector<double> &viSig) const;
		/** \brief Gradient of the log-posterior
		 *
		 * Calculates the patial derivative of the log-posterior for each element in the provided parameter vector.
		 *
		 * \param[in] viSig parameter vector
		 * \param[out] grad partial derivative (gradient) vector
		 *
		 */
		virtual void gradient(const vector<double> &viSig, vector<double> &grad) const;

	protected:
		/** \brief Pointer to data */
		const vector<double> *vY_;
		/** \brief Pointer to location parameters
		 *
		 * The vectorized matrix contains all location parameters: intercept, covariates (parameters with fixed priors or continuous predictors), and "random" (parameters with hierarchical priors).
		 */
		const vector<double> *vTheta_;
		/** \brief Pointer to vectorized matrix covariate predictors
		 *
		 * This is a vectorized matrix of covariates, currently with a fixed low-precision Gaussian prior. Analogous to fixed effects in a mixed model. The first column is the intercept.
		 */
		const vector<double> *vX_;
		/** \brief Pointer to vector of indexes connecting hierarchy levels */
		const vector<Index> *hierInd_;
	};

	/** \brief Replicated mixture model analysis
	 *
	 * Builds a daNUTS within Gibbs sampler to fit a Gaussian mixture model for replicated data on multiple traits. Takes the data and factors for paramaters, sets the initial values, and performs the sampling.
	 */
	class WrapMMM {
	public:
		/** \brief Default constructor */
		WrapMMM() {};
		/** \brief Constructor for a one-level hierarchical model
		 *
		 * Establishes the initial parameter values and the sampler kind. Input to the factor vector must be non-negative. This should be checked in the calling function.
		 *
		 * \param[in] vY vectorized data matrix
		 * \param[in] y2line factor connecting data to lines (accessions)
		 * \param[in] ln2pop factor connecting lines to populations
		 * \param[in] d number of traits
		 * \param[in] trueISig vector of true inverse-covariances (for development)
		 * \param[in] tau0 prior precision for the "fixed" effects
		 */
		WrapMMM(const vector<double> &vY, const vector<size_t> &y2line, const vector<size_t> &ln2pop, const size_t &d, const vector<double> &trueISig, const double &tau0);
		/** \brief Copy constructor (deleted) */
		WrapMMM(WrapMMM &in) = delete;
		/** \brief Move constructor (deleted) */
		WrapMMM(WrapMMM &&in) = delete;
		/** \brief Destructor */
		~WrapMMM();
		/** \brief Sampler
		 *
		 * Runs the chosen sampler with given parameters and outputs the chain.
		 *
		 * \param[in] Nadapt number of adaptation (burn-in) steps
		 * \param[in] Nsample number of sampling steps
		 * \param[out] chain MCMC chain
		 */
		void runSampler(const uint32_t &Nadapt, const uint32_t &Nsample, vector<double> &chain);
	private:
		/** \brief Data vector */
		vector<double> vY_;
		/** \brief Vector of indexes connecting hierarchy levels
		 *
		 * First element connects replicates (data) to line means, second connects lines to populations.
		 *
		 */
		vector<Index> hierInd_;
		/** \brief Location parameters */
		vector<double> vTheta_;
		/** \brief Inerese-covariances */
		vector<double> vISig_;
		/** \brief Models
		 *
		 * The location parameter model first, then the inverse-covariance model.
		 */
		vector<Model*> models_;
		/** \brief Vector of pointers to samplers
		 *
		 * Will point to the chosen derived sampler class(es).
		 */
		vector<Sampler*> sampler_;
		/** \brief Random number generator */
		RanDraw rng_;

	};
}
#endif /* lme_hpp */

