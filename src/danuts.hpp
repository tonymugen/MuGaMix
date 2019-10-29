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
 * Class definition for an implementation of the No-U-Turn Sampler with dual averaging.
 *
 */

#ifndef danuts_hpp
#define danuts_hpp

#include <vector>

#include "random.hpp"
#include "model.hpp"
#include "sampler.hpp"

using std::vector;

namespace BayesicSpace {
	//forward declaration
	class SamplerNUTS;

	/** \brief NUTS sampler class
	 *
	 * MCMC sampler class that implements the No-U-Turn Sampling with a dual-averaging algorithm to autimatically set the Hamiltonian step size \f$ \epsilon \f$. A class that implements a statistical model has to provide function to calculate a log-posterior and its gradient, as well as a pointer to the parameter vector.
	 *
	 */
	class SamplerNUTS : public Sampler {
	public:
		/** \brief Default constructor */
		SamplerNUTS() : Sampler(), theta_{nullptr} {};
		/** \brief Constructor
		 *
		 * Sets up the necessary functions and the pointer to the calling class parameter vector.
		 *
		 * \param[in] model pointer to a `Model` object that implements a particular statistical model
		 * \param[in] theta pointer to the vector of parameters
		 *
		 */
		SamplerNUTS(const Model *model, vector<double> *theta) : Sampler(), epsilon_{1.0}, nH0_{0.0}, m_{1.0}, Hprevious_{0.0}, logEpsBarPrevious_{0.0}, epsWMN_{0.0}, currW_{0.0}, firstAdapt_{true}, firstUpdate_{true}, model_{model}, theta_{theta} {};
		/** \brief Copy constructor (deleted)*/
		SamplerNUTS(const SamplerNUTS &in) = delete;
		/** \brief Copy assignement operator (deleted) */
		SamplerNUTS& operator=(const SamplerNUTS &in) = delete;
		/** \brief Move constructor
		 *
		 * \param[in] in object to be moved
		 */
		SamplerNUTS(SamplerNUTS &&in);
		/** \brief Move assignement operator
		 *
		 * \param[in] in object to be moved
		 * \return `SamplerNUTS` object
		 */
		SamplerNUTS& operator=(SamplerNUTS &&in);
		/** \brief Destructor */
		~SamplerNUTS(){ theta_ = nullptr; };

		/** \brief Get the current step size \f$ \epsilon \f$
		 *
		 * \return Current \f$ \epsilon \f$
		 */
		double getEpsilon() const {return epsilon_; };
		/** \brief Adaptation phase of the NUTS updating
		 *
		 * Uses Algorithm 6 from Hoffman and Gelman to settle on a good value for \f$ \epsilon \f$.
		 *
		 * Checks the output of the log-posterior function and throws an exception if it evaluates to `NaN` or \f$ +\infty \f$.
		 *
		 * \return Tree depth for the step
		 */
		virtual uint32_t adapt();
		/** \brief NUTS update of parameters
		 *
		 * The step size \f$ \epsilon \f$ set during the adaptation phase.
		 *
		 * Checks the output of the log-posterior function and throws an exception if it evaluates to `NaN` or \f$ +\infty \f$.
		 *
		 * \return Tree depth for the step
		 */
		virtual uint32_t update();

	protected:
		/** \brief Random numbers */
		RanDraw rng_;
		// NUTS parameters and functions
		/** \brief HMC step size parameter \f$\epsilon\f$ */
		double epsilon_;
		/** \brief The \f$ \Delta_{max} \f$ value for the NUTS sampler */
		static const double deltaMax_;
		// Parameters for choosing epsilon
		/** \brief Target acceptance rate \f$\delta\f$ */
		static const double delta_;
		/** \brief Stabilization parameter \f$t_0\f$ */
		static const double t0_;
		/** \brief Shrinkage parameter \f$\gamma\f$ */
		static const double gamma_;
		/** \brief Step size schedule power \f$-\kappa\f$ */
		static const double negKappa_;
		/** \brief Bit mask for the U{1,1} test */
		static const uint64_t mask_;
		/** \brief Shrinkage point \f$\mu\f$ */
		double mu_;
		/** \brief Store the \f$ -H(\theta^0, r^0) \f$ for each DA step here */
		double nH0_;
		// double &logEpsBarPrevious
		/** \brief  Warm-up step number */
		double m_;
		/** \brief The value \f$\bar{H}_{m-1}\f$ of the \f$H_t\f$ statistic from the previous warm-up step */
		double Hprevious_;
		/** \brief The value \f$\log \bar{\epsilon}_{m-1}\f$ of \f$ \epsilon \f$ being optimized, from the previous warm-up step */
		double logEpsBarPrevious_;
		/** \brief Current \f$ \epsilon \f$ weighted mean */
		double epsWMN_;
		/** \brief Current weight for the \f$ \epsilon \f$ weighted mean */
		double currW_;
		/** \brief Has the first adaptation step been run? */
		bool firstAdapt_;
		/** \brief Has the first post-adaptation update been run? */
		bool firstUpdate_;

		/** \brief Initialize step size
		 *
		 * Picks a reasonable initial value for the HMC/NUTS step size \f$\epsilon\f$. Uses Algorithm 4 from Hoffman and Gelman.
		 */
		void findInitialEpsilon_();
		/** \brief Single leapfrog step
		 *
		 * Takes a single leapfrog step, modifying \f$ \theta \f$ and \f$ r \f$; \f$ \epsilon \f$ can be negative, in which case the step is in the reverse direction.
		 *
		 * Checks the output of the log-posterior function and throws an exception if it evaluates to `NaN` or \f$ +\infty \f$.
		 *
		 * \param[in,out] theta the \f$ \theta \f$ vector
		 * \param[in,out] r the \f$ r \f$ vector
		 * \param[in] epsilon the step size \f$ \epsilon \f$, possibly negative
		 */
		void leapfrog_(vector<double> &theta, vector<double> &r, const double &epsilon);
		/** \brief Positive tree building function for the NUTS algorithm
		 *
		 * As described in Algorithm 3 of Hoffman and Gelman, but the positive direction only. Instead of \f$v\f$, I use a signed \f$ \epsilon \f$.
		 *
		 * \param[in] theta input parameter vector \f$\theta\f$
		 * \param[in] r input momentum variables \f$r\f$
		 * \param[in] lu log of the slice variable \f$u\f$
		 * \param[in] epsilon step size \f$ \epsilon \f$
		 * \param[in] j tree height \f$j\f$
		 * \param[in,out] thetaPlus positive direction parameter vector \f$ \theta^+ \f$
		 * \param[in,out] rPlus positive direction momentum variable vector \f$ r^+ \f$
		 * \param[in] thetaMinus negative direction parameter vector \f$ \theta^- \f$
		 * \param[in] rMinus negative direction momentum variable vector \f$ r^- \f$
		 * \param[out] thetaPrime proposed move \f$ \theta^{\prime} \f$ to \f$ \theta^m \f$
		 * \param[out] nPrime size \f$n^{\prime}\f$ of the implicit tree \f$ \mathcal{C}^{\prime} \f$
		 * \param[out] s stopping condition \f$s\f$
		 */
		void buildTreePos_(const vector<double> &theta, const vector<double> &r, const double &lu, const double &epsilon, const uint16_t &j, vector<double> &thetaPlus, vector<double> &rPlus, const vector<double> &thetaMinus, const vector<double> &rMinus, vector<double> &thetaPrime, double &nPrime, char &s);
		/** \brief Negative tree building function for the NUTS algorithm
		 *
		 * As described in Algorithm 3 of Hoffman and Gelman, but the negative direction only. Instead of \f$v\f$, I use a signed \f$ \epsilon \f$.
		 *
		 * Checks the output of the log-posterior function and throws an exception if it evaluates to `NaN` or \f$ +\infty \f$.
		 *
		 * \param[in] theta input parameter vector \f$\theta\f$
		 * \param[in] r input momentum variables \f$r\f$
		 * \param[in] lu log of the slice variable \f$u\f$
		 * \param[in] epsilon step size \f$ \epsilon \f$
		 * \param[in] j tree height \f$j\f$
		 * \param[in] thetaPlus positive direction parameter vector \f$ \theta^+ \f$
		 * \param[in] rPlus positive direction momentum variable vector \f$ r^+ \f$
		 * \param[in,out] thetaMinus negative direction parameter vector \f$ \theta^- \f$
		 * \param[in,out] rMinus negative direction momentum variable vector \f$ r^- \f$
		 * \param[out] thetaPrime proposed move \f$ \theta^{\prime} \f$ to \f$ \theta^m \f$
		 * \param[out] nPrime size \f$n^{\prime}\f$ of the implicit tree \f$ \mathcal{C}^{\prime} \f$
		 * \param[out] s stopping condition \f$s\f$
		 */
		void buildTreeNeg_(const vector<double> &theta, const vector<double> &r, const double &lu, const double &epsilon, const uint16_t &j, const vector<double> &thetaPlus, const vector<double> &rPlus, vector<double> &thetaMinus, vector<double> &rMinus, vector<double> &thetaPrime, double &nPrime, char &s);
		/** \brief Positive tree building function for the NUTS dual-averaging algorithm
		 *
		 * This is for the adaptation phase to find optimal \f$ \epsilon \f$, as described in Algorithm 6 of Hoffman and Gelman, but for the positive direction only. Instead of \f$v\f$, I use a signed \f$ \epsilon \f$. All other variables follow the notation in the paper. To be used in the warm-up phase to tune step size \f$\epsilon\f$.
		 *
		 * Checks the output of the log-posterior function and throws an exception if it evaluates to `NaN` or \f$ +\infty \f$.
		 *
		 * \param[in] theta input parameter vector \f$\theta\f$
		 * \param[in] r input momentum variables \f$r\f$
		 * \param[in] lu log of the slice variable \f$u\f$
		 * \param[in] epsilon step size \f$ \epsilon \f$
		 * \param[in] j tree height \f$j\f$
		 * \param[in,out] thetaPlus positive direction parameter vector \f$ \theta^+ \f$
		 * \param[in,out] rPlus positive direction momentum variable vector \f$ r^+ \f$
		 * \param[in] thetaMinus negative direction parameter vector \f$ \theta^- \f$
		 * \param[in] rMinus negative direction momentum variable vector \f$ r^- \f$
		 * \param[out] thetaPrime proposed move \f$ \theta^{\prime} \f$ to \f$ \theta^m \f$
		 * \param[out] nPrime size \f$n^{\prime}\f$ of the implicit tree \f$ \mathcal{C}^{\prime} \f$
		 * \param[out] s stopping condition \f$s\f$
		 * \param[out] alphaPrime acceptance probability \f$\alpha^{\prime}\f$ to be optimized
		 * \param[out] nAlphaPrime tree size after last doubling \f$n_{\alpha}^{\prime}\f$
		 */
		void buildTreePos_(const vector<double> &theta, const vector<double> &r, const double &lu, const double &epsilon, const uint16_t &j, vector<double> &thetaPlus, vector<double> &rPlus, const vector<double> &thetaMinus, const vector<double> &rMinus, vector<double> &thetaPrime, double &nPrime, char &s, double &alphaPrime, double &nAlphaPrime);
		/** \brief Negative tree building function for the NUTS dual-averaging algorithm
		 *
		 * This is for the adaptation phase to find optimal \f$ \epsilon \f$, as described in Algorithm 6 of Hoffman and Gelman, but for the negative direction only. Instead of \f$v\f$, I use a signed \f$ \epsilon \f$. All other variables follow the notation in the paper. To be used in the warm-up phase to tune step size \f$\epsilon\f$.
		 *
		 * Checks the output of the log-posterior function and throws an exception if it evaluates to `NaN` or \f$ +\infty \f$.
		 *
		 * \param[in] theta input parameter vector \f$\theta\f$
		 * \param[in] r input momentum variables \f$r\f$
		 * \param[in] lu log of the slice variable \f$u\f$
		 * \param[in] epsilon step size \f$ \epsilon \f$
		 * \param[in] j tree height \f$j\f$
		 * \param[in] thetaPlus positive direction parameter vector \f$ \theta^+ \f$
		 * \param[in] rPlus positive direction momentum variable vector \f$ r^+ \f$
		 * \param[in,out] thetaMinus negative direction parameter vector \f$ \theta^- \f$
		 * \param[in,out] rMinus negative direction momentum variable vector \f$ r^- \f$
		 * \param[out] thetaPrime proposed move \f$ \theta^{\prime} \f$ to \f$ \theta^m \f$
		 * \param[out] nPrime size \f$n^{\prime}\f$ of the implicit tree \f$ \mathcal{C}^{\prime} \f$
		 * \param[out] s stopping condition \f$s\f$
		 * \param[out] alphaPrime acceptance probability \f$\alpha^{\prime}\f$ to be optimized
		 * \param[out] nAlphaPrime tree size after last doubling \f$n_{\alpha}^{\prime}\f$
		 */
		void buildTreeNeg_(const vector<double> &theta, const vector<double> &r, const double &lu, const double &epsilon, const uint16_t &j, const vector<double> &thetaPlus, const vector<double> &rPlus, vector<double> &thetaMinus, vector<double> &rMinus, vector<double> &thetaPrime, double &nPrime, char &s, double &alphaPrime, double &nAlphaPrime);

		/** \brief Pointer to a model object
		 *
		 * Derived classes of this object implement particular statistical models.
		 */
		const Model *model_;
		/** \brief Pointer to the parameter vector
		 *
		 * Points to the parameters of the calling model class.
		 */
		vector<double> *theta_;
	};

}

#endif /* danuts_hpp */

