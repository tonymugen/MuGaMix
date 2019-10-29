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

/// Random number generation
/** \file
 * \author Anthony J. Greenberg
 * \copyright Copyright (c) 2017 Anthony J. Greenberg
 * \version 1.0
 *
 * Class implementation for facilities that generate random draws from various distributions.
 *
 */

#include <string>
#include <cstring>
#include <cstdint>
#include <cmath>

#include "random.hpp"

using std::string;

using namespace BayesicSpace;

// GenerateHR methods
uint64_t GenerateHR::ranInt() const {
	uint64_t rInt = 0;

	unsigned char ok  = '\0';
	string error;

	uint16_t keepTrying = 10; // number of tries before giving up
	while (keepTrying) {
		keepTrying--;

		asm volatile ("rdrand %0; setc %1"
						: "=r" (rInt), "=qm" (ok)
						:
						: "cc"
						);
		if (ok) {
			break;
		}
	}

	if (!ok) {
		error = "RDRAND_failed";
		throw error;
	}

	return rInt;
}

// GenerateMT static members
const uint16_t GenerateMT::n_  = 312;
const uint16_t GenerateMT::m_  = 156;
const uint64_t GenerateMT::lm_ = static_cast<uint64_t>(0xFFFFFFFF80000000);
const uint64_t GenerateMT::um_ = static_cast<uint64_t>(0x7FFFFFFF);
const uint64_t GenerateMT::b_  = static_cast<uint64_t>(0x71D67FFFEDA60000);
const uint64_t GenerateMT::c_  = static_cast<uint64_t>(0xFFF7EEE000000000);
const uint64_t GenerateMT::d_  = static_cast<uint64_t>(0x5555555555555555);
const uint32_t GenerateMT::l_  = 43;
const uint32_t GenerateMT::s_  = 17;
const uint32_t GenerateMT::t_  = 37;
const uint32_t GenerateMT::u_  = 29;
const uint64_t GenerateMT::alt_[2]  = {static_cast<uint64_t>(0), static_cast<uint64_t>(0xB5026F5AA96619E9)};

//GenerateMT methods
GenerateMT::GenerateMT(){
	// start by using RDTSC to set the seed
	uint32_t lo   = 0;
	uint32_t hi   = 0;
	uint64_t seed = 0;
	asm volatile ("rdtsc;"
					: "=a" (lo), "=d" (hi)
					);
	seed = (static_cast<uint64_t>(hi)<<32)|lo;
	uint64_t f = 6364136223846793005ULL;
	mt_[0]     = seed;
	mti_       = 1;

	for (; mti_ < n_; mti_++) {
		mt_[mti_] = (f * (mt_[mti_ - 1]^(mt_[mti_ - 1] >> 62)) + mti_);
	}

}

uint64_t GenerateMT::ranInt() const {

	if (mti_ == n_) { // do we need to re-run the twister?
		size_t i = 0;
		for (; i < n_ - m_; i++) { // first _m words
			x_     = (mt_[i]&um_)|(mt_[i + 1]&lm_);
			mt_[i] = mt_[i + m_]^(x_>>1)^alt_[static_cast<size_t>(x_&1ULL)];
		}
		for (; i < n_ - 1; i++) { // rest, except for the last element
			x_     = (mt_[i]&um_)|(mt_[i+1]&lm_);
			mt_[i] = mt_[i + (m_ - n_)] ^ (x_>>1)^alt_[static_cast<size_t>(x_&1ULL)];
		}
		// now set the last element
		x_          = (mt_[n_ - 1]&um_)|(mt_[0]&lm_);
		mt_[n_ - 1] = mt_[m_ - 1]^(x_>>1)^alt_[static_cast<size_t>(x_&1ULL)];

		mti_ = 0;

	}
	// extract pseudo-random number
	x_ = mt_[mti_++];

	x_ ^= ((x_>>u_)&d_);
	x_ ^= ((x_<<s_)&b_);
	x_ ^= ((x_<<t_)&c_);
	x_ ^= (x_>>l_);

	return x_;
}

// RanDraw static members
const double RanDraw::paramR_ = 3.44428647676;
const double RanDraw::ytab_[128] = {
	1.0, 0.963598623011, 0.936280813353, 0.913041104253,
	0.892278506696, 0.873239356919, 0.855496407634, 0.838778928349,
	0.822902083699, 0.807732738234, 0.793171045519, 0.779139726505,
	0.765577436082, 0.752434456248, 0.739669787677, 0.727249120285,
	0.715143377413, 0.703327646455, 0.691780377035, 0.68048276891,
	0.669418297233, 0.65857233912, 0.647931876189, 0.637485254896,
	0.62722199145, 0.617132611532, 0.607208517467, 0.597441877296,
	0.587825531465, 0.578352913803, 0.569017984198, 0.559815170911,
	0.550739320877, 0.541785656682, 0.532949739145, 0.524227434628,
	0.515614886373, 0.507108489253, 0.498704867478, 0.490400854812,
	0.482193476986, 0.47407993601, 0.466057596125, 0.458123971214,
	0.450276713467, 0.442513603171, 0.434832539473, 0.427231532022,
	0.419708693379, 0.41226223212, 0.404890446548, 0.397591718955,
	0.390364510382, 0.383207355816, 0.376118859788, 0.369097692334,
	0.362142585282, 0.355252328834, 0.348425768415, 0.341661801776,
	0.334959376311, 0.328317486588, 0.321735172063, 0.31521151497,
	0.308745638367, 0.302336704338, 0.29598391232, 0.289686497571,
	0.283443729739, 0.27725491156, 0.271119377649, 0.265036493387,
	0.259005653912, 0.253026283183, 0.247097833139, 0.241219782932,
	0.235391638239, 0.229612930649, 0.223883217122, 0.218202079518,
	0.212569124201, 0.206983981709, 0.201446306496, 0.195955776745,
	0.190512094256, 0.185114984406, 0.179764196185, 0.174459502324,
	0.169200699492, 0.1639876086, 0.158820075195, 0.153697969964,
	0.148621189348, 0.143589656295, 0.138603321143, 0.133662162669,
	0.128766189309, 0.123915440582, 0.119109988745, 0.114349940703,
	0.10963544023, 0.104966670533, 0.100343857232, 0.0957672718266,
	0.0912372357329, 0.0867541250127, 0.082318375932, 0.0779304915295,
	0.0735910494266, 0.0693007111742, 0.065060233529, 0.0608704821745,
	0.056732448584, 0.05264727098, 0.0486162607163, 0.0446409359769,
	0.0407230655415, 0.0368647267386, 0.0330683839378, 0.0293369977411,
	0.0256741818288, 0.0220844372634, 0.0185735200577, 0.0151490552854,
	0.0118216532614, 0.00860719483079, 0.00553245272614, 0.00265435214565
};
const uint64_t RanDraw::ktab_[128] = {
	0, 12590644, 14272653, 14988939,
	15384584, 15635009, 15807561, 15933577,
	16029594, 16105155, 16166147, 16216399,
	16258508, 16294295, 16325078, 16351831,
	16375291, 16396026, 16414479, 16431002,
	16445880, 16459343, 16471578, 16482744,
	16492970, 16502368, 16511031, 16519039,
	16526459, 16533352, 16539769, 16545755,
	16551348, 16556584, 16561493, 16566101,
	16570433, 16574511, 16578353, 16581977,
	16585398, 16588629, 16591685, 16594575,
	16597311, 16599901, 16602354, 16604679,
	16606881, 16608968, 16610945, 16612818,
	16614592, 16616272, 16617861, 16619363,
	16620782, 16622121, 16623383, 16624570,
	16625685, 16626730, 16627708, 16628619,
	16629465, 16630248, 16630969, 16631628,
	16632228, 16632768, 16633248, 16633671,
	16634034, 16634340, 16634586, 16634774,
	16634903, 16634972, 16634980, 16634926,
	16634810, 16634628, 16634381, 16634066,
	16633680, 16633222, 16632688, 16632075,
	16631380, 16630598, 16629726, 16628757,
	16627686, 16626507, 16625212, 16623794,
	16622243, 16620548, 16618698, 16616679,
	16614476, 16612071, 16609444, 16606571,
	16603425, 16599973, 16596178, 16591995,
	16587369, 16582237, 16576520, 16570120,
	16562917, 16554758, 16545450, 16534739,
	16522287, 16507638, 16490152, 16468907,
	16442518, 16408804, 16364095, 16301683,
	16207738, 16047994, 15704248, 15472926
};
const double RanDraw::wtab_[128] = {
	1.62318314817e-08, 2.16291505214e-08, 2.54246305087e-08, 2.84579525938e-08,
	3.10340022482e-08, 3.33011726243e-08, 3.53439060345e-08, 3.72152672658e-08,
	3.8950989572e-08, 4.05763964764e-08, 4.21101548915e-08, 4.35664624904e-08,
	4.49563968336e-08, 4.62887864029e-08, 4.75707945735e-08, 4.88083237257e-08,
	5.00063025384e-08, 5.11688950428e-08, 5.22996558616e-08, 5.34016475624e-08,
	5.44775307871e-08, 5.55296344581e-08, 5.65600111659e-08, 5.75704813695e-08,
	5.85626690412e-08, 5.95380306862e-08, 6.04978791776e-08, 6.14434034901e-08,
	6.23756851626e-08, 6.32957121259e-08, 6.42043903937e-08, 6.51025540077e-08,
	6.59909735447e-08, 6.68703634341e-08, 6.77413882848e-08, 6.8604668381e-08,
	6.94607844804e-08, 7.03102820203e-08, 7.11536748229e-08, 7.1991448372e-08,
	7.2824062723e-08, 7.36519550992e-08, 7.44755422158e-08, 7.52952223703e-08,
	7.61113773308e-08, 7.69243740467e-08, 7.77345662086e-08, 7.85422956743e-08,
	7.93478937793e-08, 8.01516825471e-08, 8.09539758128e-08, 8.17550802699e-08,
	8.25552964535e-08, 8.33549196661e-08, 8.41542408569e-08, 8.49535474601e-08,
	8.57531242006e-08, 8.65532538723e-08, 8.73542180955e-08, 8.8156298059e-08,
	8.89597752521e-08, 8.97649321908e-08, 9.05720531451e-08, 9.138142487e-08,
	9.21933373471e-08, 9.30080845407e-08, 9.38259651738e-08, 9.46472835298e-08,
	9.54723502847e-08, 9.63014833769e-08, 9.71350089201e-08, 9.79732621669e-08,
	9.88165885297e-08, 9.96653446693e-08, 1.00519899658e-07, 1.0138063623e-07,
	1.02247952126e-07, 1.03122261554e-07, 1.04003996769e-07, 1.04893609795e-07,
	1.05791574313e-07, 1.06698387725e-07, 1.07614573423e-07, 1.08540683296e-07,
	1.09477300508e-07, 1.1042504257e-07, 1.11384564771e-07, 1.12356564007e-07,
	1.13341783071e-07, 1.14341015475e-07, 1.15355110887e-07, 1.16384981291e-07,
	1.17431607977e-07, 1.18496049514e-07, 1.19579450872e-07, 1.20683053909e-07,
	1.21808209468e-07, 1.2295639141e-07, 1.24129212952e-07, 1.25328445797e-07,
	1.26556042658e-07, 1.27814163916e-07, 1.29105209375e-07, 1.30431856341e-07,
	1.31797105598e-07, 1.3320433736e-07, 1.34657379914e-07, 1.36160594606e-07,
	1.37718982103e-07, 1.39338316679e-07, 1.41025317971e-07, 1.42787873535e-07,
	1.44635331499e-07, 1.4657889173e-07, 1.48632138436e-07, 1.50811780719e-07,
	1.53138707402e-07, 1.55639532047e-07, 1.58348931426e-07, 1.61313325908e-07,
	1.64596952856e-07, 1.68292495203e-07, 1.72541128694e-07, 1.77574279496e-07,
	1.83813550477e-07, 1.92166040885e-07, 2.05295471952e-07, 2.22600839893e-07
};

RanDraw::RanDraw(){
	string error;

	uint32_t eax;
	uint32_t ebx;
	uint32_t ecx;
	uint32_t edx;

	uint32_t leaf    = 0;
	uint32_t subleaf = 0;

	// first look at vendor info
	asm volatile ("cpuid;"
					: "=a" (eax), "=b" (ebx), "=c" (ecx), "=d" (edx)
					: "a" (leaf), "c" (subleaf)
					);

	if ( !(
				(memcmp(reinterpret_cast<char*>(&ebx), "Genu", 4) && memcmp(reinterpret_cast<char*>(&edx), "ineI", 4) && memcmp(reinterpret_cast<char*>(&ecx), "ntel", 4)) ||
				(memcmp(reinterpret_cast<char*>(&ebx), "Auth", 4) && memcmp(reinterpret_cast<char*>(&edx), "enti", 4) && memcmp(reinterpret_cast<char*>(&ecx), "cAMD", 4))
			)) {
		error = "CPU_unsupported";
		throw error;
	}


	// now test for RDRAND. bit 30 of ECX when CPUINFO invoked with EAX set to 01H
	leaf = 0x1;
	asm volatile ("cpuid;"
					: "=a" (eax), "=b" (ebx), "=c" (ecx), "=d" (edx)
					: "a" (leaf), "c" (subleaf)
					);

	if ( (ecx & 0x40000000) == 0x40000000) { // bit mask 0x40000000 for the 30th bit
		rng_  = new GenerateHR(); // have hardware random numbers
		kind_ = 'h';
	} else {
		rng_  = new GenerateMT(); // use MT
		kind_ = 'm';
	}

}
uint64_t RanDraw::sampleInt(const uint64_t &min, const uint64_t &max) const {
	if (min >= max) {
		throw string("Lower bound not smaller than upper bound");
	}
	return min + this->ranInt()%(max - min);
}

vector<uint64_t> RanDraw::shuffleUint(const uint64_t &N){
	vector<uint64_t> out(N);
	for (uint64_t k = 0; k < N; ++k) {
		out[k] = k;
	}
	for (uint64_t i = N - 1; i > 0; --i) {
		uint64_t j = this->sampleInt(i+1); // sampleInt(n) samples i < n
		// the three XORs trick to swap to integers
		if (i != j) { // no move needed if this is actually the same variable
			out[i] ^= out[j];
			out[j] ^= out[i];
			out[i] ^= out[j];
		}
	}
	return out; // relying on copy elision
}
double RanDraw::runifnz() const {
	double rnz = 0.0;
	do {
		rnz = this->runif();
	} while (rnz == 0.0); // simply reject 0.0

	return rnz;
}

double RanDraw::runifno() const {
	double rno = 0.0;
	do {
		rno = this->runif();
	} while (rno == 1.0); // simply reject 1.0

	return rno;
}

double RanDraw::runifop() const {
	double rno = 0.0;
	do {
		rno = this->runif();
	} while ((rno == 1.0) || (rno == 0.0)); // simply reject 1.0 and 0.0

	return rno;
}

double RanDraw::rnorm() const {
	double x;
	double sign;
	uint64_t i;
	uint64_t j;

	while (1) {
		i = rng_->ranInt()&255UL;      // choose the step
		j = rng_->ranInt()%16777216UL; // sample from 2^24

		sign = (i & 0x80) ? 1.0 : -1.0;
		i &= 0x7f; // convert i to [0, 127], an index into the ziggurat slices

		x = static_cast<double>(j) * wtab_[i];
		if (j < ktab_[i]) { // _ktab is used to test for acceptance without floating point operations
			break;
		}
		double y;
		if (i < 127){
			y = ytab_[i+1] + (ytab_[i]-ytab_[i+1])*(this->runifno());
		} else {
			double U1 = this->runifnz(); // (0,1]
			double U2 = this->runifno(); // [0,1)
			x = paramR_ - log(U1)/paramR_;
			y = exp(-paramR_*(x - 0.5*paramR_)) * U2;
		}
		if ( y < exp(-0.5*x*x) ){
			break;
		}
	}

	return sign*x;
}

double RanDraw::rgamma(const double &alpha) const {
	if (alpha <= 0.0) {
		return nan("");
	}
	if (alpha < 1.0) {
		return this->rgamma(alpha + 1.0)*pow(this->runifop(), 1.0/alpha);
	}
	double x, v, u;
	double d = alpha - 0.3333333333;
	double c = 0.3333333333/sqrt(d);

	while (1) {
		do {
			x = this->rnorm();
			v = 1.0 + c*x;
		} while (v <= 0.0);

		v = v*v*v;
		u = this->runifop();

		if (u < 1.0 - 0.0331*x*x*x*x){
			break;
		}
		if (log(u) < 0.5*x*x + d*(1.0 - v + log(v))){
			break;
		}


	}

	return d*v;
}
uint64_t RanDraw::vitterA(const double &n, const double &N) const {
	// The notation follows Vitter's (1987) as closely as possible
	// Note that my runif() is on [0,1] (Vitter assumes (0,1)), so I have to sometimes adjust accordingly
	uint64_t S  = 0;
	double top  = N - n;
	double quot = top/N;
	double v;

	// some trivial conditions first
	if ( (n == 0) || (n > N) ) {
		return S;
	} else if (n == 1) {
		do {
			S = static_cast<uint64_t>(floor(N * this->runif()));

		} while (S == static_cast<uint64_t>(N)); // s == N would be a problem because it would overshoot the end of the array/file (in base-0 space); effectively sampling [0,1) uniform to prevent this
		return S;
	}
	// [0,1) uniform
	do {
		v = this->runif();
	} while (v == 1.0); // i.e. only repeat if v == 1.0

	double Nloc = N;
	while (quot > v) {
		S++;
		top--;
		Nloc--;
		quot = quot*top/Nloc;
	}

	return S;
}

uint64_t RanDraw::vitter(const double &n, const double &N) const {
	// The notation follows Vitter's (1987) as closely as possible
	// Note that my runif() is on [0,1] (Vitter assumes (0,1)), so I have to sometimes adjust accordingly
	uint64_t S = 0;
	double alphaInv = 13.0;
	if (n >= N/alphaInv) { // if the threshold is not satisfied, use Vitter's A algorithm
		return this->vitterA(n, N);
	} else if (n == 1){ // trivial case
		do {
			S = static_cast<uint64_t>(floor(N * this->runif()));

		} while (S == static_cast<uint64_t>(N)); // s == N would be a problem because it would overshoot the end of the array/file (in base-0 space); effectively sampling [0,1) uniform to prevent this
		return S;

	}

	// if we pass all thresholds, we use Vitter's rejection scheme
	double nInv     = 1.0/n;
	double nMin1inv = 1.0/(n - 1.0);
	double qu1db    = 1.0 + N - n;
	uint64_t qu1    = static_cast<uint64_t>(qu1db);
	double Vprime;
	double X;
	double Sdb;
	double U;
	double y1;

	// outer loop in Vitter (1987) A2
	while (1) {

		unsigned int d2tst = 0;
		do {  // step D2; generate U and X
			Vprime = pow(this->runif(), nInv);
			X      = N * (1.0 - Vprime);
			S      = floor(X);
			d2tst++;
		} while (S >= qu1);

		U      = this->runif();
		Sdb    = static_cast<double>(S);
		y1     = pow(U*N/qu1db, nMin1inv);
		Vprime = y1 * (1.0 - X/N) * (qu1db/(qu1db - Sdb));
		if (Vprime < 1.0) { // Step D3: accept test 2.8 (Vitter 1987)
			break;
		}
		// moving on to Step D4
		double y2  = 1.0;
		double top = N - 1.0;
		double bottom;
		double limit;
		if (Sdb < n - 1.0) {
			bottom = N - n;
			limit  = N - Sdb;
		} else {
			bottom = N - Sdb - 1.0;
			limit  = qu1db;
		}

		// calculate f(|_X_|)
		for (double t = N - 1.0; t >= limit; t--) {
			y2 = y2 * top/bottom;
			top--;
			bottom--;
		}
		if (N/(N - X) >= y1 * pow(y2, nMin1inv)) { // Accept D4 condition
			break;
		}
		// reject everything, go back to the start
	}
	return S;
}




