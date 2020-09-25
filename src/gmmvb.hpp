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
 * Class definition and interface documentation for variational Bayes inference of Gaussian mixture models.
 *
 */
#ifndef gmmvb_hpp
#define gmmvb_hpp

#include <vector>

#include "matrixView.hpp"
#include "utilities.hpp"
#include "random.hpp"
#include "index.hpp"

using std::vector;

namespace BayesicSpace {
	// forward declarations
	class GmmVB;
	class GmmVBmiss;

	/** \brief Variational Bayes class
	 *
	 * Implements Gaussian mixture model fit using variational Bayes
	 */
	class GmmVB {
	public:
		/** \brief Default constructor */
		GmmVB() : yVec_{nullptr}, N_{nullptr}, lambda0_{0.0}, nu0_{0.0}, sigmaSq0_{0.0}, alpha0_{0.0}, d_{0.0}, nu0p2_{0.0}, nu0p1_{0.0}, dln2_{0.0}, maxIt_{0}, stoppingDiff_{0.0} {};
		/** \brief Constructor
		 *
		 * The vectorized matrices must be in the column major format (as in R and FORTRAN). For larger population numbers, make sure \f$ \nu_0 > d - 2 \f$.
		 *
		 * \param[in] yVec pointer to vectorized data matrix
		 * \param[in] lambda0 prior precision scale factor
		 * \param[in] sigmaSq0 prior variance
		 * \param[in] alpha0 prior population size
		 * \param[in] nPop number of populations
		 * \param[in] d number of traits
		 * \param[in,out] vPopMn pointer to vectorized matrix of population means
		 * \param[in, out] vSm pointer to vectorized collection of population covariances
		 * \param[in, out] resp pointer to vectorized matrix responsibilities
		 * \param[in, out] Nm pointer to vector of effective population sizes
		 */
		GmmVB(const vector<double> *yVec, const double &lambda0, const double &sigmaSq0, const double alpha0, const size_t &nPop, const size_t &d, vector<double> *vPopMn, vector<double> *vSm, vector<double> *resp, vector<double> *Nm);
		/** \brief Destructor */
		~GmmVB(){ yVec_ = nullptr; N_ = nullptr; };

		/** \brief Copy constructor (deleted) */
		GmmVB(const GmmVB &in) = delete;
		/** \brief Copy assignment (deleted) */
		GmmVB& operator=(const GmmVB &in) = delete;
		/** \brief Move constructor
		 *
		 * \param[in] in object to move
		 */
		GmmVB(GmmVB &&in);
		/** \brief Move assignment operator (deleted)
		 *
		 * \param[in] in object to be moved
		 * \return target object
		 */
		GmmVB& operator=(GmmVB &&in) = delete;

		/** \brief Fit model
		 *
		 * Fits the model, returning the log-posterior for each step and the end-result deviance information criterion (DIC).
		 *
		 * \param[out] logPost vector of lower bounds
		 * \param[out] dic the DIC value
		 */
		void fitModel(vector<double> &logPost, double &dic);
	protected:
		/** \brief Pointer to vectorized data matrix */
		const vector<double> *yVec_;
		/** \brief Matrix view of the data */
		MatrixViewConst Y_;
		/** \brief Population means matrix view */
		MatrixView M_;
		/** \brief Vector of weighted covariance matrix views */
		vector<MatrixView> W_;
		/** \brief `W_` log-determinants */
		vector<double> lnDet_;
		/** \brief Sum of digammas */
		vector<double> sumDiGam_;
		/** \brief Matrix view of responsibilities */
		MatrixView R_;
		/** \brief Pointer to vector of effective population sizes */
		vector<double> *N_;

		// constants
		/** \brief Prior covariance scale factor */
		const double lambda0_;
		/** \brief Prior precision degrees of freedom */
		const double nu0_;
		/** \brief Prior variance */
		const double sigmaSq0_;
		/** \brief Prior population size */
		const double alpha0_;
		/** \brief Double version of the trait number */
		const double d_;
		/** \brief nu_0 + 2 */
		const double nu0p2_;
		/** \brief nu_0 + 1 */
		const double nu0p1_;
		/** \brief \f$ d \ln 2 \f$ */
		const double dln2_;
		/** \brief Maximum number of iterations */
		const uint16_t maxIt_;
		/** \brief Stopping criterion */
		const double stoppingDiff_;
		/** \brief Natural log of `DBL_MAX` */
		static const double lnMaxDbl_;

		// Utilities
		/** \brief Numerical utilities */
		NumerUtil nuc_;
		/** \brief Random numbers */
		RanDraw rng_;

		// Private functions
		/** \brief The E-step */
		virtual void eStep_();
		/** \brief The M-step */
		virtual void mStep_();
		/** \brief Log-posterior function
		 *
		 * Calculates the log-posterior to monitor convergence.
		 *
		 * \return the log-posterior value
		 */
		virtual double logPost_();
		/** \brief Euclidean distance between matrix rows
		 *
		 * \param[in] m1 first matrix
		 * \param[in] row1 index of the first matrix row
		 * \param[in] m2 second matrix
		 * \param[in] row2 index of the second matrix row
		 * \return euclidean distance between the rows
		 */
		virtual double rowDistance_(const MatrixViewConst &m1, const size_t &row1, const MatrixView &m2, const size_t &row2);
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
		virtual void kMeans_(const MatrixViewConst &X, const size_t &Kclust, const uint32_t &maxIt, Index &x2m, MatrixView &M);
	};
	/** \brief Variational Bayes with missing data
	 *
	 * Implements Gaussian mixture model fit using variational Bayes. Missing data are allowed.
	 *
	 */
	class GmmVBmiss final : public GmmVB {
	public:
		/** \brief Default constructor */
		GmmVBmiss() : GmmVB() {};
		/** \brief Constructor
		 *
		 * The vectorized matrices must be in the column major format (as in R and FORTRAN). Missing values must be marked with `nan("")`. For larger population numbers, make sure \f$ \nu_0 > d - 2 \f$.
		 *
		 * \param[in] yVec pointer to vectorized data matrix
		 * \param[in] lambda0 prior precision scale factor
		 * \param[in] sigmaSq0 prior variance
		 * \param[in] alpha0 prior population size
		 * \param[in] nPop number of populations
		 * \param[in] d number of traits
		 * \param[in,out] vPopMn pointer to vectorized matrix of population means
		 * \param[in, out] vSm pointer to vectorized collection of population covariances
		 * \param[in, out] resp pointer to vectorized matrix responsibilities
		 * \param[in, out] Nm pointer to vector of effective population sizes
		 */
		GmmVBmiss(const vector<double> *yVec, const double &lambda0, const double &sigmaSq0, const double alpha0, const size_t &nPop, const size_t &d, vector<double> *vPopMn, vector<double> *vSm, vector<double> *resp, vector<double> *Nm) : GmmVB(yVec, lambda0, sigmaSq0, alpha0, nPop, d, vPopMn, vSm, resp, Nm) {};
		/** \brief Destructor */
		~GmmVBmiss(){ yVec_ = nullptr; N_ = nullptr; };

		/** \brief Copy constructor (deleted) */
		GmmVBmiss(const GmmVBmiss &in) = delete;
		/** \brief Copy assignment (deleted) */
		GmmVBmiss& operator=(const GmmVBmiss &in) = delete;
		/** \brief Move constructor
		 *
		 * \param[in] in object to move
		 */
		GmmVBmiss(GmmVBmiss &&in);
		/** \brief Move assignment operator (deleted)
		 *
		 * \param[in] in object to be moved
		 * \return target object
		 */
		GmmVBmiss& operator=(GmmVBmiss &&in) = delete;
	protected:
		// Private functions
		/** \brief The E-step */
		void eStep_() override;
		/** \brief The M-step */
		void mStep_() override;
		/** \brief Log-posterior function
		 *
		 * Calculates the log-posterior to monitor convergence.
		 *
		 * \return the log-posterior value
		 */
		double logPost_() override;
		/** \brief Euclidean distance between matrix rows
		 *
		 * \param[in] m1 first matrix
		 * \param[in] row1 index of the first matrix row
		 * \param[in] m2 second matrix
		 * \param[in] row2 index of the second matrix row
		 * \return euclidean distance between the rows
		 */
		double rowDistance_(const MatrixViewConst &m1, const size_t &row1, const MatrixView &m2, const size_t &row2) override;
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
		void kMeans_(const MatrixViewConst &X, const size_t &Kclust, const uint32_t &maxIt, Index &x2m, MatrixView &M) override;
	};
}

#endif /* gmmvb_hpp */
