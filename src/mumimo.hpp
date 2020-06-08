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

/// Multiplicative mixture models
/** \file
 * \author Anthony J. Greenberg
 * \copyright Copyright (c) 2019 Anthony J. Greenberg
 * \version 1.0
 *
 * Class definition and interface documentation to generate Markov chains for inference from multiplicative Gaussian mixture models. Dual-averaging NUTS and Metropolis samplers for parameters groups are included within a Gibbs sampler.
 *
 */
#ifndef lme_hpp
#define lme_hpp

#include <vector>
#include <cmath>
#include <map>

#include "index.hpp"
#include "random.hpp"
#include "model.hpp"
#include "sampler.hpp"
#include "danuts.hpp"
#include "matrixView.hpp"

using std::vector;
using std::map;

namespace BayesicSpace {
	// forward declarations
	class MumiNR;
	class MumiPNR;
	class MumiLocNR;
	class MumiISigNR;
	class MumiLoc;
	class MumiISig;

	class WrapMMMnr;
	class WrapMMM;

	/** \brief No-replication multiplicative mixture model parameter component
	 *
	 * Models all mixture model components given population assignment probabilities. Implements the log-posterior and gradient methods.
	 *
	 */
	class MumiNR final : public Model {
	public:
		/** \brief Default constructor */
		MumiNR() : Model(), yVec_{nullptr}, tau0_{0.0}, nu0_{0.0}, invAsq_{0.0}, LaInd_{0}, TaInd_{0}, TpInd_{0}, NAnd_{0}, NPnd_{0}, nxnd_{0} {};
		/** \brief Constructor
		 *
		 * \param[in] yVec pointer to the data vector
		 * \param[in] pVec pointer to the population assignment probability vector
		 * \param[in] d number of traits
		 * \param[in] Npop number of populations
		 * \param[in] tau0 grand mean prior precision
		 * \param[in] nu0 inverse covariance prior degrees of freedom
		 * \param[in] invAsq inverse covariance prior precision
		 */
		MumiNR(const vector<double> *yVec, const vector<double> *pVec, const size_t &d, const size_t &Npop, const double &tau0, const double &nu0, const double &invAsq);
		/** \brief Destructor */
		~MumiNR(){yVec_ = nullptr; };

		/** \brief Copy constructor (deleted) */
		MumiNR(const MumiNR &in) = delete;
		/** \brief Copy assignment (deleted) */
		MumiNR& operator=(const MumiNR &in) = delete;
		/** \brief Move constructor
		 *
		 * \param[in] in object to move
		 */
		MumiNR(MumiNR &&in);
		/** \brief Move assignment operator
		 *
		 * \param[in] in object to be moved
		 * \return target object
		 */
		MumiNR& operator=(MumiNR &&in);
		/** \brief Log-posterior function
		 *
		 * Returns the value of the log-posterior given the data provided at construction and the passed-in parameter vector.
		 * The parameter vector has population means, among-individual inverse covariances, among-individual precisions, and among-population precisions in that order.
		 *
		 * \param[in] theta parameter vector
		 * \return Value of the log-posterior
		 */
		double logPost(const vector<double> &theta) const override;
		/** \brief Gradient of the log-posterior
		 *
		 * Calculates the partial derivative of the log-posterior for each element in the provided parameter vector.
		 *
		 * \param[in] theta parameter vector
		 * \param[out] grad partial derivative (gradient) vector
		 *
		 */
		void gradient(const vector<double> &theta, vector<double> &grad) const override;
	protected:
		/** \brief Pointer to the data vector */
		const vector<double> *yVec_;
		/** \brief Matrix view of data */
		MatrixViewConst Y_;
		/** \brief Population assignment probability matrix */
		MatrixViewConst P_;
		/** \brief Line factorized precision matrix view
		 *
		 * Points to `vLa_`.
		 */
		mutable MatrixView La_;
		/** \brief Expanded \f$ \boldsymbol{L}_A \f$ matrix
		 *
		 * Vectorized line unity triangular matrix (\f$\boldsymbol{L}_A\f$ in the model description).
		 */
		mutable vector<double> vLa_;
		/** \brief Grand mean prior precision */
		double tau0_;
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
		// the following indices refer to the theta passed to the log-post and gradient functions
		/** \brief First element of the among-individual inverse-covariances */
		size_t LaInd_;
		/** \brief First element of the among-individual precisions */
		size_t TaInd_;
		/** \brief First element of the among-population precisions */
		size_t TpInd_;
		/** \brief N_A + nu0 + 2d */
		double NAnd_;
		/** \brief N_P + nu0 + 2d */
		double NPnd_;
		/** \brief nu0*(nu0 + 2d) */
		double nxnd_;
		/** \brief Expand the vector of factorized precision matrices
		 *
		 * Expands the triangular \f$\boldsymbol{L}_A\f$ matrix contained in the precision matrix potion of the parameter vector into the internal `L_` vector.
		 * The parameter vector stores only the non-zero elements of these matrix.
		 *
		 * \param[in] theta the parameter vector
		 */
		void expandISvec_(const vector<double> &theta) const;
	};

	/** \brief Population assignement probability component
	 *
	 * Models population assignment probabilities given current values of parameters for the rest of the unreplicated hierarchical model.
	 * Implements log-posterior and gradient methods.
	 *
	 */
	class MumiPNR final : public Model {
	public:
		/** \brief Default constructor */
		MumiPNR() : Model(), yVec_{nullptr}, alphaPr_{0.0}, theta_{nullptr}, LaInd_{0}, TaInd_{0}, TpInd_{0} {};
		/** \brief Constructor
		 *
		 * \param[in] yVec pointer to the data vector
		 * \param[in] theta pointer to hierarchical model parameters
		 * \param[in] d number of traits
		 * \param[in] Npop number of populations
		 * \param[in] alphaPr prior population proportions
		 */
		MumiPNR(const vector<double> *yVec, const vector<double> *theta, const size_t &d, const size_t &Npop, const double &alphaPr);
		/** \brief Destructor */
		~MumiPNR(){yVec_ = nullptr; };

		/** \brief Copy constructor (deleted) */
		MumiPNR(const MumiPNR &in) = delete;
		/** \brief Copy assignment (deleted) */
		MumiPNR& operator=(const MumiPNR &in) = delete;
		/** \brief Move constructor
		 *
		 * \param[in] in object to move
		 */
		MumiPNR(MumiPNR &&in);
		/** \brief Move assignment operator
		 *
		 * \param[in] in object to be moved
		 * \return target object
		 */
		MumiPNR& operator=(MumiPNR &&in);
		/** \brief Log-posterior function
		 *
		 * Returns the value of the log-posterior given the data provided at construction and the passed-in parameter vector.
		 * The parameter vector has population means, among-individual inverse covariances, among-individual precisions, and among-population precisions in that order.
		 *
		 * \param[in] vPhi vectorized matrix of transformed population assignment probabilities
		 * \return Value of the log-posterior
		 */
		double logPost(const vector<double> &vPhi) const override;
		/** \brief Gradient of the log-posterior
		 *
		 * Calculates the partial derivative of the log-posterior for each element in the provided parameter vector.
		 *
		 * \param[in] vPhi vectorized matrix of transformed population assignment probabilities
		 * \param[out] grad partial derivative (gradient) vector
		 *
		 */
		void gradient(const vector<double> &vPhi, vector<double> &grad) const override;

	protected:
		/** \brief Pointer to vectorized data */
		const vector<double> *yVec_;
		/** \brief Matrix view of data */
		MatrixViewConst Y_;
		/** \brief Prior population proportions
		 *
		 * To save computation, actually stores \f$ \alpha - 1 \f$
		 *
		 */
		double alphaPr_;
		/** \brief Pointer to model data */
		const vector<double> *theta_;
		/** Population mean matrix view */
		MatrixViewConst M_;
		/** Overall mean matrix view */
		MatrixViewConst mu_;
		/** \brief Line factorized precision matrix view
		 *
		 * Points to `vLa_`.
		 */
		mutable MatrixView La_;
		/** \brief Expanded \f$ \boldsymbol{L}_A \f$ matrix
		 *
		 * Vectorized line unity triangular matrix (\f$\boldsymbol{L}_A\f$ in the model description).
		 */
		mutable vector<double> vLa_;
		// the following indices refer to the theta passed to the log-post and gradient functions
		/** \brief First element of the among-individual inverse-covariances */
		size_t LaInd_;
		/** \brief First element of the among-individual precisions */
		size_t TaInd_;
		/** \brief First element of the among-population precisions */
		size_t TpInd_;
		/** \brief Expand the vector of factorized precision matrices
		 *
		 * Expands the triangular \f$\boldsymbol{L}_A\f$ matrix contained in the precision matrix potion of the parameter vector into the internal `L_` vector.
		 * The parameter vector stores only the non-zero elements of these matrix.
		 *
		 */
		void expandISvec_() const;

	};
	/** \brief Mixture model for location parameters, no replication
	 *
	 * Implements log-posterior and gradient for the location parameters of the multiplicative mixture model with no replication.
	 *
	 */
	class MumiLocNR final : public Model {
	public:
		/** \brief Default constructor */
		MumiLocNR() : Model(), tau0_{0.0}, iSigTheta_{nullptr}, fTaInd_{0}, Npop_{0}, alphaPr_{1.0} {};
		/** \brief Constructor
		 *
		 * \param[in] yVec pointer vectorized data matrix
		 * \param[in] d number of traits
		 * \param[in] iSigVec pointer to vectorized inverse-covariance matrix collection
		 * \param[in] xVec pointer to vectorized covariate predictor matrix
		 * \param[in] tau fixed prior for the unmodeled ("fixed") effects and overall mean (intercept)
		 * \param[in] nPops number of populations
		 * \param[in] alphaPr prior on \f$ \alpha \f$ on population assignment probabilities
		 */
		MumiLocNR(const vector<double> *yVec, const size_t &d, const vector<double> *iSigVec, const double &tau, const size_t &nPops, const double &alphaPr);
		/** \brief Destructor */
		~MumiLocNR(){iSigTheta_ = nullptr; yVec_ = nullptr; };

		/** \brief Copy constructor (deleted) */
		MumiLocNR(const MumiLocNR &in) = delete;
		/** \brief Copy assignment (deleted) */
		MumiLocNR& operator=(const MumiLocNR &in) = delete;
		/** \brief Move constructor
		 *
		 * \param[in] in object to move
		 */
		MumiLocNR(MumiLocNR &&in);
		/** \brief Move assignment operator
		 *
		 * \param[in] in object to be moved
		 * \return target object
		 */
		MumiLocNR& operator=(MumiLocNR &&in);
		/** \brief Log-posterior function
		 *
		 * Returns the value of the log-posterior given the data provided at construction and the passed-in parameter vector. The parameter vector has the population means and population assignment probabilities in that order.
		 *
		 * \param[in] theta parameter vector
		 * \return Value of the log-posterior
		 */
		double logPost(const vector<double> &theta) const override;
		/** \brief Gradient of the log-posterior
		 *
		 * Calculates the partial derivative of the log-posterior for each element in the provided parameter vector.
		 *
		 * \param[in] theta parameter vector
		 * \param[out] grad partial derivative (gradient) vector
		 *
		 */
		void gradient(const vector<double> &theta, vector<double> &grad) const override;

	protected:
		/** \brief Pointer to the data vector */
		const vector<double> *yVec_;
		/** \brief Matrix view of data */
		MatrixViewConst Y_;
		/** \brief Fixed prior precision for unmodeled effects */
		double tau0_;
		/** \brief Pointer to a precision parameter vector */
		const vector<double> *iSigTheta_;
		/** \brief Line factorized precision matrix view
		 *
		 * Points to `vLa_`.
		 */
		mutable MatrixView La_;
		/** \brief Expanded \f$ \boldsymbol{L}_A \f$ matrix
		 *
		 * Vectorized line unity triangular matrix (\f$\boldsymbol{L}_A\f$ in the model description).
		 */
		mutable vector<double> vLa_;
		// Constants
		/** \brief Index of the first \f$\boldsymbol{T}_A\f$ element */
		size_t fTaInd_;
		/** \brief Index of the first \f$\boldsymbol{T}_P\f$ element */
		size_t fTpInd_;
		/** \brief Index of the first probability element */
		size_t PhiBegInd_;
		/** \brief Number of populations */
		size_t Npop_;
		/** \brief Prior population assignment probability \f$\alpha_0 - 1\f$ */
		double alphaPr_;
		/** \brief Expand the vector of factorized precision matrices
		 *
		 * Expands the triangular \f$\boldsymbol{L}_X\f$ matrices contained in the precision matrix vector into the internal `L_` vector.
		 * The input vector stores only the non-zero elements of these matrices.
		 *
		 */
		void expandISvec_() const;
	};
	/** \brief Mixture model for location parameters
	 *
	 * Implements log-posterior and gradient for the location parameters of the multiplicative mixture model with replicated observations.
	 *
	 */
	class MumiLoc final : public Model {
	public:
		/** \brief Default constructor */
		MumiLoc() : Model(), hierInd_{nullptr}, tau0_{0.0}, iSigTheta_{nullptr}, fTeInd_{0}, fLaInd_{0}, fTaInd_{0}, Npop_{0}, tauPrPhi_{1.0}, alphaPr_{1.0} {};
		/** \brief Constructor
		 *
		 * \param[in] yVec pointer vectorized data matrix
		 * \param[in] iSigVec pointer to vectorized inverse-covariance matrix collection
		 * \param[in] xVec pointer to vectorized covariate predictor matrix
		 * \param[in] hierInd pointer to vector of hierarchical indexes
		 * \param[in] tau fixed prior for the unmodeled ("fixed") effects and overall mean (intercept)
		 * \param[in] nPops number of populations
		 * \param[in] tauPrPhi \f$ \tau_{\phi} \f$ population assignment probability prior precision
		 * \param[in] alphaPr prior on \f$ \alpha \f$ on population assignment probabilities
		 */
		MumiLoc(const vector<double> *yVec, const vector<double> *iSigVec, const vector<Index> *hierInd, const double &tau, const size_t &nPops, const double &tauPrPhi, const double &alphaPr);
		/** \brief Destructor */
		~MumiLoc(){hierInd_ = nullptr; iSigTheta_ = nullptr; };

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
		double logPost(const vector<double> &theta) const override;
		/** \brief Gradient of the log-posterior
		 *
		 * Calculates the partial derivative of the log-posterior for each element in the provided parameter vector.
		 *
		 * \param[in] theta parameter vector
		 * \param[out] grad partial derivative (gradient) vector
		 *
		 */
		void gradient(const vector<double> &theta, vector<double> &grad) const override;

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
		/** \brief Line factorized precision matrix view
		 *
		 * Points to `vLx_`.
		 */
		mutable MatrixView La_;
		/** \brief Expanded \f$ \boldsymbol{L} \f$ matrices
		 *
		 * Vectorized error and line unity triangular matrices (\f$\boldsymbol{L}_X\f$ in the model description).
		 */
		mutable vector<double> vLx_;
		// Constants
		/** \brief Cut-off for \f$ p_{jm} \f$ approximation */
		static const double pSumCutOff_;
		/** \brief Index of the first \f$\boldsymbol{T}_E\f$ element */
		size_t fTeInd_;
		/** \brief Index of the first \f$\boldsymbol{L}_A\f$ element */
		size_t fLaInd_;
		/** \brief Index of the first \f$\boldsymbol{T}_A\f$ element */
		size_t fTaInd_;
		/** \brief Index of the first \f$\boldsymbol{T}_P\f$ element */
		size_t fTpInd_;
		/** \brief Index of the first probability element */
		size_t PhiBegInd_;
		/** \brief Number of populations */
		size_t Npop_;
		/** \brief The \f$ \tau_{\phi} \f$ prior precision*/
		double tauPrPhi_;
		/** \brief Prior population assignment probability */
		double alphaPr_;
		/** \brief Expand the vector of factorized precision matrices
		 *
		 * Expands the triangular \f$\boldsymbol{L}_X\f$ matrices contained in the precision matrix vector into the internal `L_` vector.
		 * The input vector stores only the non-zero elements of these matrices.
		 *
		 */
		void expandISvec_() const;
	};

	/** \brief Model for inverse covariances with no replication
	 *
	 * Implements log-posterior and gradient for inverse covariances for the multiplicative mixture model with no replication.
	 * The inverse-covariances are factorized and stored compactly in the vectors provided to the methods of this class.
	 * The unit lower-triangular \f$\boldsymbol{L}_A\f$ is stored first (by column and excluding the diagonal), then the diagonal log-precision matrix \f$\boldsymbol{T}_A\f$, then \f$\boldsymbol{T}_M\f$ (see the model description for notation).
	 *
	 */
	class MumiISigNR final : public Model {
	public:
		/** \brief Default constructor */
		MumiISigNR(): Model(), nu0_{2.0}, invAsq_{1e-10}, fTaInd_{0}, fTpInd_{0} {};
		/** \brief Constructor
		 *
		 * \param[in] yVec pointer to data
		 * \param[in] d number of traits
		 * \param[in] vTheta pointer to vector of location parameters
		 * \param[in] xVec pointer to vectorized covariate matrix (with intercept)
		 * \param[in] nu0 prior degrees of freedom \f$\nu_0\f$
		 * \param[in] invAsq prior precision \f$a^{-2}\f$
		 * \param[in] nPops number of populations
		 *
		 */
		MumiISigNR(const vector<double> *yVec, const size_t &d, const vector<double> *vTheta, const double &nu0, const double &invAsq, const size_t &nPops);

		/** \brief Destructor */
		~MumiISigNR(){ };
		/** \brief Copy constructor (deleted) */
		MumiISigNR(const MumiLocNR &in) = delete;
		/** \brief Copy assignment (deleted) */
		MumiISigNR& operator=(const MumiISigNR &in) = delete;
		/** \brief Move constructor
		 *
		 * \param[in] in object to move
		 */
		MumiISigNR(MumiISigNR &&in);
		/** \brief Move assignment operator
		 *
		 * \param[in] in object to be moved
		 * \return target object
		 */
		MumiISigNR& operator=(MumiISigNR &&in);
		/** \brief Log-posterior function
		 *
		 * Returns the value of the log-posterior given the data provided at construction and the passed-in parameter vector.
		 *
		 * \param[in] viSig parameter vector
		 * \return Value of the log-posterior
		 */
		double logPost(const vector<double> &viSig) const override;
		/** \brief Gradient of the log-posterior
		 *
		 * Calculates the partial derivative of the log-posterior for each element in the provided parameter vector.
		 *
		 * \param[in] viSig parameter vector
		 * \param[out] grad partial derivative (gradient) vector
		 *
		 */
		void gradient(const vector<double> &viSig, vector<double> &grad) const override;

	protected:
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
		/** \brief Covariate effect view */
		MatrixViewConst B_;
		/** \brief Population mean view */
		MatrixViewConst Mp_;
		/** \brief Overall mean view */
		MatrixViewConst mu_;
		/** \brief Population assignment logit-probability view */
		MatrixViewConst Phi_;

		/** \brief Line factorized precision matrix view
		 *
		 * Points to `vLa_`.
		 */
		mutable MatrixView La_;
		/** \brief Expanded \f$ \boldsymbol{L}_A \f$ matrix
		 *
		 * Vectorized line unity triangular matrix (\f$\boldsymbol{L}_A\f$ in the model description).
		 */
		mutable vector<double> vLa_;
		// Constants
		/** \brief Index of the first \f$\boldsymbol{T}_A\f$ element */
		size_t fTaInd_;
		/** \brief Index of the first \f$\boldsymbol{T}_P\f$ element */
		size_t fTpInd_;
		/** \brief nu0*(nu0 + 2d) */
		double nxnd_;
		/** \brief N_A + nu0 + 2d */
		double NAnd_;
		/** \brief N_P + nu0 + 2d */
		double NPnd_;
		/** \brief Expand the vector of factorized precision matrices
		 *
		 * Expands the triangular \f$\boldsymbol{L}_A\f$ matrix contained in the provided vector into the internal `L_` vector. The input vector stores only the non-zero elements of these matrices.
		 *
		 * \param[in] viSig compressed vector of factorized precision matrices
		 */
		void expandISvec_(const vector<double> &viSig) const;
	};
	/** \brief Model for inverse covariances
	 *
	 * Implements log-posterior and gradient for inverse covariances. The inverse-covariances are factorized and stored compactly in the vectors provided to the methods of this class.
	 * The error matrix is stored first, then the line precision matrix. The unit lower-triangular \f$\boldsymbol{L}_X\f$ is stored first (by column and excluding the diagonal), then the diagonal log-precision matrix \f$\boldsymbol{T}_X\f$ (see the model description for notation).
	 *
	 */
	class MumiISig final : public Model {
	public:
		/** \brief Default constructor */
		MumiISig(): Model(), hierInd_{nullptr}, nu0_{2.0}, invAsq_{1e-10}, fTeInd_{0}, fTaInd_{0}, fTpInd_{0} {};
		/** \brief Constructor
		 *
		 * \param[in] yVec pointer to data
		 * \param[in] vTheta pointer to vector of location parameters
		 * \param[in] xVec pointer to vectorized covariate matrix (with intercept)
		 * \param[in] hierInd pointer to a vector with hierarchical indexes
		 * \param[in] nu0 prior degrees of freedom \f$\nu_0\f$
		 * \param[in] invAsq prior precision \f$a^{-2}\f$
		 * \param[in] nPops number of populations
		 *
		 */
		MumiISig(const vector<double> *yVec, const vector<double> *vTheta, const vector<Index> *hierInd, const double &nu0, const double &invAsq, const size_t &nPops);

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
		double logPost(const vector<double> &viSig) const override;
		/** \brief Gradient of the log-posterior
		 *
		 * Calculates the partial derivative of the log-posterior for each element in the provided parameter vector.
		 *
		 * \param[in] viSig parameter vector
		 * \param[out] grad partial derivative (gradient) vector
		 *
		 */
		void gradient(const vector<double> &viSig, vector<double> &grad) const override;

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
		/** \brief Covariate effect view */
		MatrixViewConst B_;
		/** \brief Population mean view */
		MatrixViewConst Mp_;
		/** \brief Overall mean view */
		MatrixViewConst mu_;
		/** \brief Population assignment logit-probability view */
		MatrixViewConst Phi_;

		/** \brief Error factorized precision matrix view
		 *
		 * Points to `vLx_`.
		 */
		mutable MatrixView Le_;
		/** \brief Line factorized precision matrix view
		 *
		 * Points to `vLx_`.
		 */
		mutable MatrixView La_;
		/** \brief Expanded \f$ \boldsymbol{L} \f$ matrices
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
	 * Builds a daNUTS within Gibbs sampler to fit a Gaussian mixture model for replicated data on multiple traits. Takes the data and factors for parameters, sets the initial values, and performs the sampling.
	 */
	class WrapMMM {
	public:
		/** \brief Default constructor */
		WrapMMM() {};
		/** \brief Constructor for a model with no replication
		 *
		 * Establishes the initial parameter values and the sampler kind.
		 *
		 * \param[in] vY vectorized data matrix
		 * \param[in] d number of traits
		 * \param[in] Npop number of populations
		 * \param[in] alphaPr \f$\alpha \f$ prior parameter for population assignment probabilities
		 * \param[in] tau0 prior precision for the "fixed" effects
		 * \param[in] nu0 prior degrees of freedom for precision matrices
		 * \param[in] invAsq prior inverse variance for precision matrices
		 */
		WrapMMM(const vector<double> &vY, const size_t &d, const uint32_t &Npop, const double &alphaPr, const double &tau0, const double &nu0, const double &invAsq);
		/** \brief Constructor for a one-level hierarchical model
		 *
		 * Establishes the initial parameter values and the sampler kind. Input to the factor vector must be non-negative. This should be checked in the calling function.
		 *
		 * \param[in] vY vectorized data matrix
		 * \param[in] y2line factor connecting data to lines (accessions)
		 * \param[in] Npop number of populations
		 * \param[in] tauPrPhi prior precision for \f$\phi \f$
		 * \param[in] alphaPr \f$\alpha \f$ prior parameter for population assignment probabilities
		 * \param[in] tau0 prior precision for the "fixed" effects
		 * \param[in] nu0 prior degrees of freedom for precision matrices
		 * \param[in] invAsq prior inverse variance for precision matrices
		 */
		WrapMMM(const vector<double> &vY, const vector<size_t> &y2line, const uint32_t &Npop, const double &tauPrPhi, const double &alphaPr, const double &tau0, const double &nu0, const double &invAsq);
		/** \brief Constructor for a one-level hierarchical model with missing data
		 *
		 * Establishes the initial parameter values and the sampler kind. Input to the factor vector must be non-negative. This should be checked in the calling function.
		 *
		 * \param[in] vY vectorized data matrix
		 * \param[in] y2line factor connecting data to lines (accessions)
		 * \param[in] missIDs vectorized matrix (same dimensions as `vY`) with 1 corresponding to a missing data point and 0 otherwise
		 * \param[in] Npop number of populations
		 * \param[in] tauPrPhi prior precision for \f$\phi \f$
		 * \param[in] alphaPr \f$\alpha \f$ prior parameter for population assignment probabilities
		 * \param[in] tau0 prior precision for the "fixed" effects
		 * \param[in] nu0 prior degrees of freedom for precision matrices
		 * \param[in] invAsq prior inverse variance for precision matrices
		 */
		WrapMMM(const vector<double> &vY, const vector<size_t> &y2line, const vector<int32_t> &missIDs, const uint32_t &Npop, const double &tauPrPhi, const double &alphaPr, const double &tau0, const double &nu0, const double &invAsq);
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
		 * \param[out] thetaChain MCMC chain of location parameters
		 * \param[out] isigChain MCMC chain of inverse-covariance parameters
		 * \param[out] piChain MCMC chain of \f$ p_{ip} \f$
		 */
		void runSampler(const uint32_t &Nadapt, const uint32_t &Nsample, const uint32_t &Nthin, vector<double> &thetaChain, vector<double> &isigChain, vector<double> &piChain);
		/** \brief Sampler with missing data
		 *
		 * Runs the sampler with given parameters, imputes missing data, and outputs chains. Imputed values for the missing data points are in the `impYchain` variable.
		 * The imputed data are arranged by row first, then by trait index within the row.
		 *
		 * \param[in] Nadapt number of adaptation (burn-in) steps
		 * \param[in] Nsample number of sampling steps
		 * \param[in] Nthin thinning number
		 * \param[out] thetaChain MCMC chain of model parameters
		 * \param[out] isigChain MCMC chain of inverse-covariance parameters
		 * \param[out] piChain MCMC chain of \f$ p_{ip} \f$
		 * \param[out] impYchain MCMC chain tracking imputed data
		 */
		void runSampler(const uint32_t &Nadapt, const uint32_t &Nsample, const uint32_t &Nthin, vector<double> &thetaChain, vector<double> &isigChain, vector<double> &piChain, vector<double> &impYchain);
	protected:
		/** \brief Vectorized data matrix
		 *
		 * Vectorized matrix of responses.
		 */
		vector<double> vY_;
		/** \brief Data matrix view
		 *
		 * Points to `vY_`.
		 */
		MatrixView Y_;
		/** \brief Vector of indexes connecting hierarchy levels
		 *
		 * First element connects replicates (data) to line means, second connects lines to populations. The second index is updated as part of the mixture model.
		 *
		 */
		vector<Index> hierInd_;
		/** \brief Model paramaters */
		vector<double> vTheta_;
		/** \brief Inverse-covariances */
		vector<double> vISig_;
		/** \brief Transformed population assignment probabilities */
		vector<double> vPhi_;
		/** \brief Population assignment probabilities */
		vector<double> vP_;
		/** \brief Matrix view of line (accession) means */
		MatrixView A_;
		/** \brief Matrix view of population means */
		MatrixView Mp_;
		/** \brief Index of the first probability element */
		size_t PhiBegInd_;
		/** \brief Matrix view of logit-population assignment probabilities */
		MatrixView Phi_;
		/** \brief Matrix view of population assignment probabilies */
		MatrixView P_;
		/** \brief `Phi_` recalibration trigger value */
		static const double phiMin_;
		/** \brief Value to add for calibration */
		static const double addVal_;
		/** \brief Index of the first \f$\boldsymbol{T}_E\f$ element */
		size_t fTeInd_;
		/** \brief Index of the first \f$\boldsymbol{L}_A\f$ element */
		size_t fLaInd_;
		/** \brief Index of the first \f$\boldsymbol{T}_A\f$ element */
		size_t fTaInd_;
		/** \brief Expanded vectorized \f$ \boldsymbol{L}_A \f$ matrix */
		vector<double> vLa_;
		/** \brief Matrix view of the \f$ \boldsymbol{L}_A \f$ matrix */
		MatrixView La_;
		/** \brief Vectorized \f$ \boldsymbol{A} - \boldsybol{\mu}_{p\cdot} \f$ residual */
		vector<double> vAresid_;
		/** \brief Matrix view of the residual */
		MatrixView Aresid_;
		/** \brief First principal component of \f$\boldsymbol{\Sigma}_A\f$
		 *
		 * The first principal component vector of the initial estimate of the line (accession) covariance matrix, to be used for projection ordering of population means.
		 */
		vector<double> pc1_;
		/** \brief Missingness index
		 *
		 * The key values are indexes of rows that have missing data, the mapped value is a vector with indexes of missing phenotypes for that row.
		 */
		map<size_t, vector<size_t> > missInd_;
		/** \brief Models
		 *
		 * The location parameter model first, then the inverse-covariance model.
		 */
		vector<Model*> models_;
		/** \brief Vector of pointers to samplers
		 *
		 * Will point to the chosen derived sampler class(es).
		 */
		vector<Sampler*> samplers_;
		/** \brief Random number generator */
		RanDraw rng_;
		/** \brief Impute missing phenotype data */
		void imputeMissing_();
		/** \brief Sort the populations
		 *
		 * Populations are sorted using a projection on the first PC of the initial line covariance matrix (`pc1_`). The rows in the population matrix `Mp_` are then re-arranged accordingly (in order of increase). The elements of the `pi_` mixture proportion vector are also rearranged.
		 */
		void sortPops_();
		/** \brief Calibrate rows of `Phi_`
		 *
		 * Looks for rows of `Phi_` with all elements negative and below a threshold (currently - 5.0, determined empirically) and adds a constant to increase numerical stability.
		 * This does not affect population assignment probabilities.
		 */
		void calibratePhi_();
		/** \brief Convert probabilities to logit-scores
		 *
		 * Uses the Betancourt (2012) method of transforming population assignment probabilities to hyper-spherical coordinates. I then apply a logit transformation to further map scores to the \f$(-\infty, \infty ) \f$ interval.
		 *
		 */
		void p2phi_();
		/** \brief Expand lower triangle of the \f$ \boldsymbol{L}_A \f$ matrix
		 *
		 * Expands the triangular \f$\boldsymbol{L}_A\f$ matrix and multiplies its columns by the square root of \f$ \boldsymbol{T}_A \f$. The input vector `vISig_` stores only the non-zero elements of these matrices.
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
		double rowDistance_(const MatrixView &m1, const size_t &row1, const MatrixView &m2, const size_t &row2);
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

