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

	/** \brief Shell sort
	 *
	 * Sorts the provided vector in ascending order using Shell's method. Rather than move the elements themselves, save their indexes to the output vector. The first element of the index vector points to the smallest element of the input vector etc. The implementation is modified from code in Numerical Recipes in C++.
	 * NOTE: This algorithm is too slow for vectors of \f$ > 50\f$ elements. I am using it for population projection ordering, where the number of populations will typically not exceed 10.
	 *
	 * \param[in] target vector to be sorted
	 * \param[in] beg index of the first element
	 * \param[in] end index of one past the last element to be included
	 * \param[out] outIdx vector of indexes
	 */
	void shellSort(const vector<double> &target, const size_t &beg, const size_t &end, vector<size_t> &outIdx){
#ifndef PKG_DEBUG_OFF
		if (target.size() < end) {
			throw string("Target vector size smaller than end index in shellSort()");
		} else if (outIdx.size() < end) {
			throw string("Output vector size smaller than end index in shellSort()");
		} else if (end < beg) {
			throw string("End index smaller than beginning index in shellSort()");
		} else if (target.size() != outIdx.size()) {
			throw string("Target and output vectors must be of the same size in shellSort()");
		}
#endif
		// pick the initial increment
		size_t inc = 1;
		do {
			inc = inc*3 + 1;
		} while (inc <= end - beg);

		// start the sort
		do { // loop over partial sorts, decreasing the increment each time
			inc /= 3;
			const size_t bottom = beg + inc;
			for (size_t iOuter = bottom; iOuter < end; iOuter++) { // outer loop of the insertion sort, going over the indexes
				if (outIdx[iOuter] >= target.size()) {
					throw string("outIdx value out of bounds for target vector in shellSort()");
				}
				const size_t curInd = outIdx[iOuter]; // save the current value of the index
				size_t jInner       = iOuter;
				while (target[ outIdx[jInner - inc] ] > target[ curInd ]) {  // Straight insertion inner loop; looking for a place to insert the current value
					if (outIdx[jInner-inc] >= target.size()) {
						throw string("outIdx value out of bounds for target vector in shellSort()");
					}
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

	/** \brief Mixture model for location parameters
	 *
	 * Implements log-posterior and gradient for the location parameters of the mixture model.
	 *
	 */
	class MumiLoc : public Model {
	public:
		/** \brief Default constructor */
		MumiLoc() : Model(), hierInd_{nullptr}, tau0_{0.0}, iSigTheta_{nullptr}, fTeInd_{0}, fLaInd_{0}, fTaInd_{0} {};
		/** \brief Constructor
		 *
		 * \param[in] yVec pointer vectorized data matrix
		 * \param[in] iSigVec pointer to vectorized inverse-covariance matrix collection
		 * \param[in] xVec pointer to vectorized covariate predictor matrix
		 * \param[in] hierInd pointer to vector of hierarchical indexes
		 * \param[in] tau fixed prior for the unmodeled ("fixed") effects and population means
		 */
		MumiLoc(const vector<double> *yVec, const vector<double> *iSigVec, const vector<Index> *hierInd, const double &tau);
		/** \brief Destructor */
		~MumiLoc(){hierInd_ = nullptr; };

		/** \brief Copy constructor (deleted) */
		MumiLoc(const MumiLoc &in) = delete;
		/** \brief Copy assignment (deleted) */
		MumiLoc& operator=(const MumiLoc &in) = delete;
		/** \brief Move constructor
		 *
		 * \param[in] in object to move
		 */
		MumiLoc(MumiLoc &&in);
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
		/** \brief Pointer to vector of indexes connecting hierarchy levels */
		const vector<Index> *hierInd_;
		/** \brief Fixed prior precision for unmodeled effects */
		double tau0_;
		/** \brief Pointer to a precision parameter vector */
		const vector<double> *iSigTheta_;
		/** \brief Error factorized precision matrix view
		 *
		 * Points to `vLx_`.
		 */
		mutable MatrixView Le_;
		/** \brief Line factorized preficision matrix view
		 *
		 * Points to `vLx_`.
		 */
		mutable MatrixView La_;
		/** \brief Expanded _L_ matrices
		 *
		 * Vectorized error and line unity triangular matrices (\f$\boldsymbol{L}_X\f$ in the model description).
		 */
		mutable vector<double> vLx_;
		// Constants
		/** \brief Index of the first \f$\boldsymbol{T}_E\f$ element */
		size_t fTeInd_;
		/** \brief Index of the first \f$\boldsymbol{L}_A\f$ element */
		size_t fLaInd_;
		/** \brief Index of the first \f$\boldsymbol{T}_A\f$ element */
		size_t fTaInd_;
		/** \brief Index of the first \f$\boldsymbol{T}_P\f$ element */
		size_t fTpInd_;
		/** \brief Expand the vector of factorized precision matrices
		 *
		 * Expands the triangular \f$\boldsymbol{L}_X\f$ matrices contained in the precision matrix vector into the internal `L_` vector. The input vector stores only the non-zero elements of these matrices.
		 *
		 */
		void expandISvec_() const;
	};
	/** \brief Model for inverse covariances
	 *
	 * Implements log-posterior and gradient for inverse covariances. The inverse-covariances are factorized and stored compactly in the vectors provided to the methods of this class.
	 * The error matrix is stored first, then the line precision matrix. The unit lower-triangular \f$\boldsymbol{L}_X\f$ is stored first (by column and excluding the diagonal), then the diagonal log-precision matrix \f$\boldsymbol{T}_X\f$ (see the model description for notation).
	 *
	 */
	class MumiISig : public Model {
	public:
		/** \brief Default constructor */
		MumiISig(): Model(), hierInd_{nullptr}, nu0_{2.0}, invAsq_{1e-10}, fTeInd_{0}, fTaInd_{0} {};
		/** \brief Constructor
		 *
		 * \param[in] yVec pointer to data
		 * \param[in] vTheta pointer to vector of location parameters
		 * \param[in] xVec pointer to vectorized covariate matrix (with intercept)
		 * \param[in] hierInd pointer to a vector with hierarchical indexes
		 * \param[in] nu0 prior degrees of freedom \f$\nu_0\f$
		 * \param[in] invAsq prior precision \f$a^{-2}\f$
		 *
		 */
		MumiISig(const vector<double> *yVec, const vector<double> *vTheta, const vector<Index> *hierInd, const double &nu0, const double &invAsq);

		/** \brief Destructor */
		~MumiISig(){hierInd_ = nullptr; };
		/** \brief Copy constructor (deleted) */
		MumiISig(const MumiLoc &in) = delete;
		/** \brief Copy assignment (deleted) */
		MumiISig& operator=(const MumiISig &in) = delete;
		/** \brief Move constructor
		 *
		 * \param[in] in object to move
		 */
		MumiISig(MumiISig &&in);
		/** \brief Move assignment operator
		 *
		 * \param[in] in object to be moved
		 * \return target object
		 */
		MumiISig& operator=(MumiISig &&in);
		/** \brief Log-posterior function
		 *
		 * Returns the value of the log-posterior given the data provided at construction and the passed-in parameter vector.
		 *
		 * \param[in] viSig parameter vector
		 * \return Value of the log-posterior
		 */
		double logPost(const vector<double> &viSig) const;
		/** \brief Gradient of the log-posterior
		 *
		 * Calculates the patial derivative of the log-posterior for each element in the provided parameter vector.
		 *
		 * \param[in] viSig parameter vector
		 * \param[out] grad partial derivative (gradient) vector
		 *
		 */
		void gradient(const vector<double> &viSig, vector<double> &grad) const;

	protected:
		/** \brief Pointer to vector of indexes connecting hierarchy levels */
		const vector<Index> *hierInd_;
		/** \brief Prior degrees of freedom
		 *
		 * Degrees of freedom of the half-\f$t\f$ prior distribution on the covariance matrix. If \f$\nu_0 = 2\f$, the prior is half-Cauchy. Should not be large for a vague prior.
		 */
		double nu0_;
		/** \brief Prior inverse-variance
		 *
		 * Inverse variance of the prior. Should be set to a large value for a vague prior.
		 */
		double invAsq_;

		/** \brief Data view */
		MatrixViewConst Y_;
		/** \brief Line mean view */
		MatrixViewConst A_;
		/** \brief Covriate effect view */
		MatrixViewConst B_;
		/** \brief Population mean view */
		MatrixViewConst Mp_;
		/** \brief Overall mean view */
		MatrixViewConst mu_;

		/** \brief Error factorized precision matrix view
		 *
		 * Points to `vLx_`.
		 */
		mutable MatrixView Le_;
		/** \brief Line factorized preficision matrix view
		 *
		 * Points to `vLx_`.
		 */
		mutable MatrixView La_;
		/** \brief Expanded _L_ matrices
		 *
		 * Vectorized error and line unity triangular matrices (\f$\boldsymbol{L}_X\f$ in the model description).
		 */
		mutable vector<double> vLx_;
		// Constants
		/** \brief Index of the first \f$\boldsymbol{T}_E\f$ element */
		size_t fTeInd_;
		/** \brief Index of the first \f$\boldsymbol{L}_A\f$ element */
		size_t fLaInd_;
		/** \brief Index of the first \f$\boldsymbol{T}_A\f$ element */
		size_t fTaInd_;
		/** \brief Index of the first \f$\boldsymbol{T}_P\f$ element */
		size_t fTpInd_;
		/** \brief nu0*(nu0 + 2d) */
		double nxnd_;
		/** \brief N + nu0 + 2d */
		double Nnd_;
		/** \brief N_A + nu0 + 2d */
		double NAnd_;
		/** \brief N_P + nu0 + 2d */
		double NPnd_;
		/** \brief Expand the vector of factorized precision matrices
		 *
		 * Expands the triangular \f$\boldsymbol{L}_X\f$ matrices contained in the provided vector into the internal `L_` vector. The input vector stores only the non-zero elements of these matrices.
		 *
		 * \param[in] viSig compressed vector of factorized precision matrices
		 */
		void expandISvec_(const vector<double> &viSig) const;
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
		 * \param[in] Npop number of populations
		 * \param[in] alphaPr prior on mixture proporions (prior proportions are assumed equal)
		 * \param[in] tau0 prior precision for the "fixed" effects
		 * \param[in] nu0 prior degrees of freedom for precision matrices
		 * \param[in] invAsq prior inverse variance for precision matrices
		 */
		WrapMMM(const vector<double> &vY, const vector<size_t> &y2line, const uint32_t &Npop, const double &alphaPr, const double &tau0, const double &nu0, const double &invAsq);
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
		 * \param[in] Nthin thinning number
		 * \param[out] thetaChain MCMC chain of model parameters
		 * \param[out] piChain MCMC chain of \f$ p_{ij} \f$
		 */
		void runSampler(const uint32_t &Nadapt, const uint32_t &Nsample, const uint32_t &Nthin, vector<double> &thetaChain, vector<double> &piChain);
	private:
		/** \brief Vectorized data matrix
		 *
		 * Vectorized matrix of responses.
		 */
		vector<double> vY_;
		/** \brief Vector of indexes connecting hierarchy levels
		 *
		 * First element connects replicates (data) to line means, second connects lines to populations. The second index is updated as part of the mixture model.
		 *
		 */
		vector<Index> hierInd_;
		/** \brief Prior mixture proportions */
		vector<double> alpha_;
		/** \brief Dirichlet expected values */
		vector<double> Dmn_;
		/** \brief Mixture proportions */
		vector<double> pi_;
		/** \brief Allocation vector */
		vector<size_t> z_;
		/** \brief Location parameters */
		vector<double> vTheta_;
		/** \brief Inerese-covariances */
		vector<double> vISig_;
		/** \brief Matrix view of line (accession) means */
		MatrixView A_;
		/** \brief Matrix view of population means */
		MatrixView Mp_;
		/** \brief Index of the first \f$\boldsymbol{L}_A\f$ element */
		size_t fLaInd_;
		/** \brief Index of the first \f$\boldsymbol{T}_A\f$ element */
		size_t fTaInd_;
		/** \brief Expanded vectorized \f$ L_A \f$ matrix */
		vector<double> vLa_;
		/** \brief Matrix view of the \f$ L_A \f$ matrix */
		MatrixView La_;
		/** \brief Vectorized \f$ A - m_{i\cdot} \f$ residual */
		vector<double> vAresid_;
		/** \brief Matrix view of the residual */
		MatrixView Aresid_;
		/** \brief Vectorized matrix of line probabilities
		 *
		 * Columns correspond to lines (accessions), rows have probabilities that a line belongs to each population, or the probability that \f$ z_i = j \f$.
		 */
		vector<double> vPz_;
		/** \brief First principal component of \f$\boldsymbol{\Sigma}_A\f$
		 *
		 * The first prinicpal component vector of the initial estimate of the line (accesion) covariance matrix, to be used for projection ordering of population means.
		 */
		vector<double> pc1_;
		/** \brief Matrix view of the probability vector */
		MatrixView Pz_;
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
		/** \brief Update mixture proportions */
		void updatePi_();
		/** \brief Update \f$ p_{ij} \f$ */
		void updatePz_();
		/** \brief Sort the populations
		 *
		 * Populations are sorted using a projection on the first PC of the initial line covariance matrix (`pc1_`). The rows in the population are then re-arranged accordingly (in order of increase).
		 */
		void sortPops_();
		/** \brief Expand lower triangle of the \f$ L_A \f$ matrix
		 *
		 * Expands the triangular \f$\boldsymbol{L}_A\f$ matrix and multiplies its columns by the square root of \f$ T_A \f$. The input vector `vISig_` stores only the non-zero elements of these matrices.
		 *
		 */
		void expandLa_();
		/** \brief Euclidean distance between matrix rows
		 *
		 * \param[in] m1 first matrix
		 * \param[in] row1 index of the first matrix row
		 * \param[in] m2 second matrix
		 * \param[in] row2 index of the second matrix row
		 * \return euclidean distance between the rows
		 */
		double rowDist_(const MatrixView &m1, const size_t &row1, const MatrixView &m2, const size_t &row2);
		/** \brief K-means clustering
		 *
		 * Performs k-means clustering on a matrix of values. Each row of the input matrix is an item with observed values in columns.
		 *
		 * \param[in] X matrix of observations to be clustered
		 * \param[in] Kclust number of clusters
		 * \param[in] maxIt maximum number of iterations
		 * \param[out] x2m `Index` relating clusters to values
		 * \param[out] M matrix of cluster means (clusters in rows)
		 */
		void kMeans_(const MatrixView &X, const size_t &Kclust, const uint32_t &maxIt, Index &x2m, MatrixView &M);
	};
}
#endif /* lme_hpp */

