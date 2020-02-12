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


/// C++ matrix class that wraps pointers.
/** \file
 * \author Anthony J. Greenberg
 * \copyright Copyright (c) 2019 Anthony J. Greenberg
 * \version 0.1
 *
 * This is the class implementation file for the experimental MatrixView class. This version is for including in R packages, so it uses the R BLAS and LAPACK interfaces.
 *
 *
 */

#include <vector>
#include <string>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <climits>
#include <utility>
#include <R_ext/Lapack.h>
#include <R_ext/BLAS.h>

#include "matrixView.hpp"
#include "random.hpp"
#include "index.hpp"

using std::vector;
using std::string;
using std::stod;
using std::fill;
using std::memcpy;
using std::nan;
using std::numeric_limits;
using std::move;

using namespace BayesicSpace;


// MatrixView methods

MatrixView::MatrixView(vector<double> *inVec, const size_t &idx, const size_t &nrow, const size_t &ncol) : data_{inVec}, idx_{idx}, Nrow_{nrow}, Ncol_{ncol} {
#ifndef PKG_DEBUG_OFF
	if (data_->size() < idx_ + Nrow_*Ncol_) {
		throw string("MatrixView indexes extend past vector end");
	}
#endif
};

MatrixView::MatrixView(MatrixView &&inMat){
	if (this != &inMat) {
		data_ = inMat.data_;
		idx_  = inMat.idx_;
		Nrow_ = inMat.Nrow_;
		Ncol_ = inMat.Ncol_;

		inMat.data_ = nullptr;
		inMat.idx_  = 0;
		inMat.Nrow_ = 0;
		inMat.Ncol_ = 0;
	}
}

MatrixView& MatrixView::operator=(MatrixView &&inMat){
	if (this != &inMat) {
		data_ = inMat.data_;
		idx_  = inMat.idx_;
		Nrow_ = inMat.Nrow_;
		Ncol_ = inMat.Ncol_;

		inMat.data_ = nullptr;
		inMat.idx_  = 0;
		inMat.Nrow_ = 0;
		inMat.Ncol_ = 0;
	}

	return *this;
}

double MatrixView::getElem(const size_t& iRow, const size_t &jCol) const{
#ifndef PKG_DEBUG_OFF
	if ((iRow >= Nrow_) || (jCol >= Ncol_)) {
		throw string("ERROR: element out of range in getElem()");
	}
#endif

	return data_->data()[idx_ + Nrow_*jCol + iRow];
}

void MatrixView::setElem(const size_t& iRow, const size_t &jCol, const double &input){
#ifndef PKG_DEBUG_OFF
	if ((iRow >= Nrow_) || (jCol >= Ncol_)) {
		throw string("ERROR: element out of range in setElem()");
	}
#endif

	data_->data()[idx_ + Nrow_*jCol + iRow] = input;

}

void MatrixView::setCol(const size_t jCol, const vector<double> data){
#ifndef PKG_DEBUG_OFF
	if (jCol >= Ncol_) {
		throw string("ERROR: column index out of range in setCol()");
	}
	if (data.size() < Nrow_) {
		throw string("ERROR: vector length smaller than the number of rows in setCol()");
	}
#endif
	double *colBeg = data_->data() + idx_ + jCol*Nrow_;
	memcpy(colBeg, data.data(), Nrow_*sizeof(double));

}

void MatrixView::chol(){
#ifndef PKG_DEBUG_OFF
	if (Nrow_ != Ncol_) {
		throw string("ERROR: matrix has to be symmetric for Cholesky decomposition");
	}
	if (Nrow_ > INT_MAX) {
		throw string("ERROR: matrix dimension too big to safely convert to int in in-place Cholesky decomposition");
	}
	if ( (Nrow_ == 0) || (Ncol_ == 0) ) {
		throw string("ERROR: one of the dimensions is zero");
	}
#endif
	int info = 0;
	char tri = 'L';

	int N = static_cast<int>(Nrow_); // conversion should be OK: magnitude of Nrow_ checked in the constructor
	dpotrf_(&tri, &N, data_->data() + idx_, &N, &info);
	if (info < 0) {
		throw string("ERROR: illegal element in in-place Cholesky decomposition");
	} else if (info > 0) {
		throw string("ERROR: matrix is not positive definite in in-place Cholesky decomposition");
	}

}

void MatrixView::chol(MatrixView &out) const {
#ifndef PKG_DEBUG_OFF
	if (Nrow_ != Ncol_) {
		throw string("ERROR: matrix has to be symmetric for Cholesky decomposition");
	}
	if ( (Nrow_ == 0) || (Ncol_ == 0) ) {
		throw string("ERROR: one of the dimensions is zero");
	}
	if ( (Nrow_ != out.Nrow_) || (Ncol_ != out.Ncol_) ) {
		throw string("ERROR: wrong dimensions in output matrix in copy Cholesky decomposition");
	}
#endif

	memcpy(out.data_->data() + out.idx_, data_->data() + idx_, (Nrow_ * Ncol_)*sizeof(double));

	int info = 0;
	char tri = 'L';

	int N = static_cast<int>(Nrow_); // conversion should be safe: Nrow_ magnitude checked during construction
	dpotrf_(&tri, &N, out.data_->data() + idx_, &N, &info);
	if (info < 0) {
		throw string("ERROR: illegal matrix element in copy Cholesky decomposition");
	} else if (info > 0) {
		throw string("ERROR: matrix is not positive definite in copy Cholesky decomposition");
	}

}

void MatrixView::cholInv(){
#ifndef PKG_DEBUG_OFF
	if (Nrow_ != Ncol_) {
		throw string("ERROR: matrix has to be symmetric for Cholesky inversion");
	}
	if ( (Nrow_ == 0) || (Ncol_ == 0) ) {
		throw string("ERROR: one of the dimensions is zero");
	}
#endif
	int info = 0;
	char tri = 'L';

	int N = static_cast<int>(Nrow_); // conversion should be safe: Nrow_ magnitude checked during construction
	dpotri_(&tri, &N, data_->data() + idx_, &N, &info);
	if (info < 0) {
		throw string("ERROR: illegal matrix element in-place Cholesky inversion");
	} else if (info > 0) {
		throw string("ERROR: a diagonal element of the matrix is zero. Cannot complete in-place Cholesky inversion");
	}
	// copying the lower triangle to the upper
	for (size_t iRow = 0; iRow < Nrow_; iRow++) {
		for (size_t jCol = 0; jCol < iRow; jCol++) {
			(*data_)[idx_ + Nrow_*iRow + jCol] = (*data_)[idx_ + Nrow_*jCol + iRow];
		}
	}

}

void MatrixView::cholInv(MatrixView &out) const {
#ifndef PKG_DEBUG_OFF
	if (Nrow_ != Ncol_) {
		throw string("ERROR: matrix has to be square for Cholesky inversion");
	}
	if ( (Nrow_ == 0) || (Ncol_ == 0) ) {
		throw string("ERROR: one of the dimensions is zero");
	}
	if ( (Nrow_ != out.Nrow_) || (Ncol_ != out.Ncol_) ) {
		throw string("ERROR: wrong dimensions in output matrix in copy Cholesky inversion");
	}
#endif

	memcpy(out.data_->data() + out.idx_, data_->data() + idx_, (Nrow_ * Ncol_)*sizeof(double));

	int info = 0;
	char tri = 'L';

	int N = static_cast<int>(Nrow_); // safe to convert: Nrow_ checked at construction
	dpotri_(&tri, &N, out.data_->data() + idx_, &N, &info);
	if (info < 0) {
		throw string("ERROR: illegal matrix element in copy Cholesky inversion");
	} else if (info > 0) {
		throw string("ERROR: a diagonal element of the matrix is zero. Cannot complete copy Cholesky inversion");
	}
	for (size_t iRow = 0; iRow < Nrow_; iRow++) {
		for (size_t jCol = 0; jCol < iRow; jCol++) {
			(*out.data_)[out.idx_ + Nrow_*iRow + jCol] = (*out.data_)[out.idx_ + Nrow_*jCol + iRow];
		}
	}
}

void MatrixView::svd(MatrixView &U, vector<double> &s){
#ifndef PKG_DEBUG_OFF
	if ( (Nrow_ == 0) || (Ncol_ == 0) ) {
		throw string("ERROR: one of the dimensions is zero");
	}
	if ((Nrow_ != U.Nrow_) || (U.Nrow_ != U.Ncol_)) {
		throw string("ERROR: wrong dimensions of the U matrix in svd()");
	}
#endif

	if (s.size() < Ncol_) {
		s.resize(Ncol_, 0.0);
	}
	int Nvt = 1;
	vector<double>vt(1, 0.0);
	int resSVD = 0;
	int Nw = -1;    // set this to pre-run dgesvd_ for calculation of workspace size
	vector<double>workArr(1, 0.0);
	char jobu  = 'A';
	char jobvt = 'N';
	// the following casts are safe because dimensions are checked at construction
	int Nr = static_cast<int>(Nrow_);
	int Nc = static_cast<int>(Ncol_);

	// first calculate working space
	dgesvd_(&jobu, &jobvt, &Nr, &Nc, data_->data()+idx_, &Nr, s.data(), U.data_->data()+U.idx_, &Nr, vt.data(), &Nvt, workArr.data(), &Nw, &resSVD);
	Nw = workArr[0];
	workArr.resize(Nw, 0.0);
	dgesvd_(&jobu, &jobvt, &Nr, &Nc, data_->data()+idx_, &Nr, s.data(), U.data_->data()+U.idx_, &Nr, vt.data(), &Nvt, workArr.data(), &Nw, &resSVD);
	workArr.resize(0);
	if (resSVD < 0) {
		throw string("ERROR: illegal matrix element in SVD");
	} else if (resSVD > 0){
		throw string("ERROR: DBDSQR did not converge in SVD");
	}

}

void MatrixView::svdSafe(MatrixView &U, vector<double> &s) const {
#ifndef PKG_DEBUG_OFF
	if ( (Nrow_ == 0) || (Ncol_ == 0) ) {
		throw string("ERROR: one of the dimensions is zero");
	}
	if ((Nrow_ != U.Nrow_) || (U.Nrow_ != U.Ncol_)) {
		throw string("ERROR: wrong dimensions of the U matrix in svdSafe()");
	}
#endif

	double *dataCopy = new double[Nrow_ * Ncol_];
	memcpy(dataCopy, data_->data()+idx_, (Nrow_ * Ncol_)*sizeof(double));

	if (s.size() < Ncol_) {
		s.resize(Ncol_, 0.0);
	}
	int Nvt = 1;
	vector<double>vt(1, 0.0);
	int resSVD = 0;
	int Nw = -1;    // set this to pre-run dgesvd_ for calculation of workspace size
	vector<double>workArr(1, 0.0);
	char jobu  = 'A';
	char jobvt = 'N';
	// the folloeing casts are safe because the dimensions are checked at construction
	int Nr = static_cast<int>(Nrow_);
	int Nc = static_cast<int>(Ncol_);

	// first calculate working space
	dgesvd_(&jobu, &jobvt, &Nr, &Nc, dataCopy, &Nr, s.data(), U.data_->data()+U.idx_, &Nr, vt.data(), &Nvt, workArr.data(), &Nw, &resSVD);
	Nw = workArr[0];
	workArr.resize(Nw, 0.0);
	dgesvd_(&jobu, &jobvt, &Nr, &Nc, dataCopy, &Nr, s.data(), U.data_->data()+U.idx_, &Nr, vt.data(), &Nvt, workArr.data(), &Nw, &resSVD);
	workArr.resize(0);
	if (resSVD < 0) {
		throw string("ERROR: illegal matrix element in safe SVD");
	} else if (resSVD > 0){
		throw string("ERROR: DBDSQR did not converge in safe SVD");
	}
	delete [] dataCopy;
}

void MatrixView::eigen(const char &tri, MatrixView &U, vector<double> &lam){
#ifndef PKG_DEBUG_OFF
	if (Nrow_ != Ncol_) {
		throw string("ERROR: matrix has to be at least square in eigen()");
	}
	if ( (Nrow_ == 0) || (Ncol_ == 0) ) {
		throw string("ERROR: one of the dimensions is zero");
	}
	if ((Ncol_ > U.Nrow_) || (Ncol_ > U.Ncol_)) {
		throw string("ERROR: wrong U matrix dimensions in eigen");
	}
#endif

	// test the output size and adjust if necessary
	if (Ncol_ > lam.size()) {
		lam.resize(Ncol_, 0.0);
	}

	char jobz  = 'V'; // computing eigenvectors
	char range = 'A'; // doing all of them
	char uplo;
	if (tri == 'u') {
		uplo = 'U';
	} else if (tri == 'l'){
		uplo = 'L';
	} else {
		throw string("ERROR: unknown triangle indicator in eigen()");
	}
	// the following casts are safe because Nrow_ magnitude is checked at construction
	int N   = static_cast<int>(Nrow_);
	int lda = static_cast<int>(Nrow_);
	// placeholder variables. Not referenced since we are computing all eigenvectors
	double vl = 0.0;
	double vu = 0.0;
	int il = 0;
	int iu = 0;

	double abstol = sqrt(numeric_limits<double>::epsilon()); // absolute tolerance. Shouldn't be too close to epsilon since I don't need very precise estimation of small eigenvalues

	int M   = N;
	int ldz = N;

	vector<int> isuppz(2*M, 0);
	vector<double> work(1, 0.0);        // workspace; size will be determined
	int lwork = -1;          // to start; this lets us determine workspace size
	vector<int> iwork(1, 0); // integer workspace; size to be calculated
	int liwork = -1;         // to start; this lets us determine integer workspace size
	int info = 0;

	dsyevr_(&jobz, &range, &uplo, &N, data_->data()+idx_, &lda, &vl, &vu, &il, &iu, &abstol, &M, lam.data(), U.data_->data()+U.idx_, &ldz, isuppz.data(), work.data(), &lwork, iwork.data(), &liwork, &info);

	lwork  = work[0];
	work.resize(static_cast<size_t>(lwork), 0.0);
	liwork = iwork[0];
	iwork.resize(static_cast<size_t>(liwork), 0);

	// run the actual estimation
	dsyevr_(&jobz, &range, &uplo, &N, data_->data()+idx_, &lda, &vl, &vu, &il, &iu, &abstol, &M, lam.data(), U.data_->data()+U.idx_, &ldz, isuppz.data(), work.data(), &lwork, iwork.data(), &liwork, &info);

	// set tiny eigenvalues to exactly zero
	for (auto &l : lam) {
		if (fabs(l) <= abstol) {
			l = 0.0;
		}
	}

}

void MatrixView::eigen(const char &tri, const size_t &n, MatrixView &U, vector<double> &lam){
#ifndef PKG_DEBUG_OFF
	if (Nrow_ != Ncol_) {
		throw string("ERROR: matrix has to be at least square in eigen()");
	}
	if (Nrow_ < n) {
		throw string("ERROR: the input number of eigenvalues greater than matrix dimensions");
	}
	if ( (Nrow_ == 0) || (Ncol_ == 0) ) {
		throw string("ERROR: one of the dimensions is zero");
	}
#endif

	if (Nrow_ == n) { // if we are doing all of them, just run regular eigen()
		MatrixView::eigen(tri, U, lam);
		return;
	}


	char jobz  = 'V'; // computing eigenvectors
	char range = 'I'; // doing some of them
	char uplo;
	if (tri == 'u') {
		uplo = 'U';
	} else if (tri == 'l'){
		uplo = 'L';
	} else {
		throw string("ERROR: unknown triangle indicator in eigen()");
	}
	int N   = static_cast<int>(Nrow_);
	int lda = static_cast<int>(Nrow_);
	// placeholder variables. Not referenced since we are computing a certain number of eigenvectors, not based on the values of the eigenvalues
	double vl = 0.0;
	double vu = 0.0;
	int il = N - static_cast<int>(n) + 1; // looks like the count base-1
	int iu = N; // do all the remaining eigenvalues

	double abstol = sqrt(numeric_limits<double>::epsilon()); // absolute tolerance. Shouldn't be too close to epsilon since I don't need very precise estimation of small eigenvalues

	int M   = iu - il + 1;
	int ldz = N;

	// test the output size and adjust if necessary
	if ((Nrow_ > U.Nrow_) || (static_cast<size_t>(M) > U.Ncol_)) {
		throw string("ERROR: wrong U matrix dimensions in eigen()");
	}
	if (static_cast<size_t>(M) > lam.size()) {
		lam.resize(static_cast<size_t>(M), 0.0);
	}

	vector<int> isuppz(2*M, 0);
	vector<double> work(1, 0.0);        // workspace; size will be determined
	int lwork = -1;          // to start; this lets us determine workspace size
	vector<int> iwork(1, 0); // integer workspace; size to be calculated
	int liwork = -1;         // to start; this lets us determine integer workspace size
	int info = 0;

	dsyevr_(&jobz, &range, &uplo, &N, data_->data()+idx_, &lda, &vl, &vu, &il, &iu, &abstol, &M, lam.data(), U.data_->data()+U.idx_, &ldz, isuppz.data(), work.data(), &lwork, iwork.data(), &liwork, &info);

	lwork  = work[0];
	work.resize(static_cast<size_t>(lwork), 0.0);
	liwork = iwork[0];
	iwork.resize(static_cast<size_t>(liwork), 0);

	// run the actual estimation
	dsyevr_(&jobz, &range, &uplo, &N, data_->data()+idx_, &lda, &vl, &vu, &il, &iu, &abstol, &M, lam.data(), U.data_->data()+U.idx_, &ldz, isuppz.data(), work.data(), &lwork, iwork.data(), &liwork, &info);

	// set tiny eigenvalues to exactly zero
	for (auto &l : lam) {
		if (fabs(l) <= abstol) {
			l = 0.0;
		}
	}

}

void MatrixView::eigenSafe(const char &tri, MatrixView &U, vector<double> &lam) const{
#ifndef PKG_DEBUG_OFF
	if (Nrow_ != Ncol_) {
		throw string("ERROR: matrix has to be at least square in eigen()");
	}
	if ( (Nrow_ == 0) || (Ncol_ == 0) ) {
		throw string("ERROR: one of the dimensions is zero");
	}
	if ((Ncol_ > U.Nrow_) || (Ncol_ > U.Ncol_)) {
		throw string("ERROR: wrong U matrix dimensions in eigenSafe()");
	}
#endif

	// test the output size and adjust if necessary
	if (Ncol_ > lam.size()) {
		lam.resize(Ncol_, 0.0);
	}

	char jobz  = 'V'; // computing eigenvectors
	char range = 'A'; // doing all of them
	char uplo;
	if (tri == 'u') {
		uplo = 'U';
	} else if (tri == 'l'){
		uplo = 'L';
	} else {
		throw string("ERROR: unknown triangle indicator in eigen()");
	}
	int N   = static_cast<int>(Nrow_);
	int lda = static_cast<int>(Nrow_);
	// placeholder variables. Not referenced since we are computing all eigenvectors
	double vl = 0.0;
	double vu = 0.0;
	int il = 0;
	int iu = 0;

	double abstol = sqrt(numeric_limits<double>::epsilon()); // absolute tolerance. Shouldn't be too close to epsilon since I don't need very precise estimation of small eigenvalues

	int M   = N;
	int ldz = N;

	vector<int> isuppz(2*M, 0);
	vector<double> work(1, 0.0);        // workspace; size will be determined
	int lwork = -1;          // to start; this lets us determine workspace size
	vector<int> iwork(1, 0); // integer workspace; size to be calculated
	int liwork = -1;         // to start; this lets us determine integer workspace size
	int info = 0;

	double *dataCopy = new double[Nrow_ * Ncol_];
	memcpy(dataCopy, data_->data()+idx_, (Nrow_ * Ncol_)*sizeof(double));

	dsyevr_(&jobz, &range, &uplo, &N, dataCopy, &lda, &vl, &vu, &il, &iu, &abstol, &M, lam.data(), U.data_->data()+U.idx_, &ldz, isuppz.data(), work.data(), &lwork, iwork.data(), &liwork, &info);

	lwork  = work[0];
	work.resize(static_cast<size_t>(lwork), 0.0);
	liwork = iwork[0];
	iwork.resize(static_cast<size_t>(liwork), 0);

	// run the actual estimation
	dsyevr_(&jobz, &range, &uplo, &N, dataCopy, &lda, &vl, &vu, &il, &iu, &abstol, &M, lam.data(), U.data_->data()+U.idx_, &ldz, isuppz.data(), work.data(), &lwork, iwork.data(), &liwork, &info);

	delete [] dataCopy;

	// set tiny eigenvalues to exactly zero
	for (auto &l : lam) {
		if (fabs(l) <= abstol) {
			l = 0.0;
		}
	}

}

void MatrixView::eigenSafe(const char &tri, const size_t &n, MatrixView &U, vector<double> &lam) const{
#ifndef PKG_DEBUG_OFF
	if (Nrow_ != Ncol_) {
		throw string("ERROR: matrix has to be at least square in eigen()");
	}
	if (Nrow_ < n) {
		throw string("ERROR: the input number of eigenvalues greater than matrix dimensions");
	}
	if ( (Nrow_ == 0) || (Ncol_ == 0) ) {
		throw string("ERROR: one of the dimensions is zero");
	}
#endif

	if (Nrow_ == n) { // if we are doing all of them, just run regular eigen()
		MatrixView::eigenSafe(tri, U, lam);
		return;
	}


	char jobz  = 'V'; // computing eigenvectors
	char range = 'I'; // doing some of them
	char uplo;
	if (tri == 'u') {
		uplo = 'U';
	} else if (tri == 'l'){
		uplo = 'L';
	} else {
		throw string("ERROR: unknown triangle indicator in eigen()");
	}
	int N   = static_cast<int>(Nrow_);
	int lda = static_cast<int>(Nrow_);
	// placeholder variables. Not referenced since we are computing a certain number of eigenvectors, not based on the values of the eigenvalues
	double vl = 0.0;
	double vu = 0.0;
	int il = N - static_cast<int>(n) + 1; // looks like the count base-1
	int iu = N; // do all the remaining eigenvalues

	double abstol = sqrt(numeric_limits<double>::epsilon()); // absolute tolerance. Shouldn't be too close to epsilon since I don't need very precise estimation of small eigenvalues

	int M   = iu - il + 1;
	int ldz = N;

	// test the output size and adjust if necessary
	if ((Nrow_ > U.Nrow_) || (static_cast<size_t>(M) > U.Ncol_)) {
		throw string("ERROR: wrong output matrix dimensions in eigenSafe()");
	}
	if (static_cast<size_t>(M) > lam.size()) {
		lam.resize(static_cast<size_t>(M), 0.0);
	}

	vector<int> isuppz(2*M, 0);
	vector<double> work(1, 0.0);  // workspace; size will be determined
	int lwork = -1;               // to start; this lets us determine workspace size
	vector<int> iwork(1, 0);      // integer workspace; size to be calculated
	int liwork = -1;              // to start; this lets us determine integer workspace size
	int info = 0;

	double *dataCopy = new double[Nrow_ * Ncol_];
	memcpy(dataCopy, data_->data()+idx_, (Nrow_ * Ncol_)*sizeof(double));

	dsyevr_(&jobz, &range, &uplo, &N, dataCopy, &lda, &vl, &vu, &il, &iu, &abstol, &M, lam.data(), U.data_->data()+U.idx_, &ldz, isuppz.data(), work.data(), &lwork, iwork.data(), &liwork, &info);

	lwork  = work[0];
	work.resize(static_cast<size_t>(lwork), 0.0);
	liwork = iwork[0];
	iwork.resize(static_cast<size_t>(liwork), 0);

	// run the actual estimation
	dsyevr_(&jobz, &range, &uplo, &N, dataCopy, &lda, &vl, &vu, &il, &iu, &abstol, &M, lam.data(), U.data_->data()+U.idx_, &ldz, isuppz.data(), work.data(), &lwork, iwork.data(), &liwork, &info);

	delete [] dataCopy;

	// set tiny eigenvalues to exactly zero
	for (auto &l : lam) {
		if (fabs(l) <= abstol) {
			l = 0.0;
		}
	}

}

void MatrixView::syrk(const char &tri, const double &alpha, const double &beta, MatrixView &C) const {
#ifndef PKG_DEBUG_OFF
	if ((Ncol_ > INT_MAX) || (Nrow_ > INT_MAX)) {
		throw string("ERROR: at least one matrix dimension too big to safely convert to int in syrk()");
	}
	if ( (Nrow_ == 0) || (Ncol_ == 0) ) {
		throw string("ERROR: one of the dimensions is zero");
	}
	if ((C.getNrows() != Ncol_) || (C.getNcols() != Ncol_)) {
		throw string("ERROR: wrong dimensions of the C matrix in syrk()");
	}
#endif

	// integer parameters
	const int n   = static_cast<int>(Ncol_);
	const int k   = static_cast<int>(Nrow_);
	const int lda = static_cast<int>(Nrow_);
	const int ldc = static_cast<int>(Ncol_);

	// transpose token
	const char trans = 't';

	dsyrk_(&tri, &trans, &n, &k, &alpha, data_->data()+idx_, &lda, &beta, C.data_->data()+C.idx_, &ldc);

}

void MatrixView::tsyrk(const char &tri, const double &alpha, const double &beta, MatrixView &C) const {
#ifndef PKG_DEBUG_OFF
	if ((Ncol_ > INT_MAX) || (Nrow_ > INT_MAX)) {
		throw string("ERROR: at least one matrix dimension too big to safely convert to int in tsyrk()");
	}
	if ( (Nrow_ == 0) || (Ncol_ == 0) ) {
		throw string("ERROR: one of the dimensions is zero");
	}
	if ((C.getNrows() != Nrow_) || (C.getNcols() != Nrow_)) {
		throw string("ERROR: wrong C matrix dimensions in tsyrk()");
	}
#endif

	// integer parameters
	const int n   = static_cast<int>(Nrow_);
	const int k   = static_cast<int>(Ncol_);
	const int lda = static_cast<int>(Nrow_);
	const int ldc = static_cast<int>(Nrow_);

	// transpose token
	const char trans = 'n';

	dsyrk_(&tri, &trans, &n, &k, &alpha, data_->data()+idx_, &lda, &beta, C.data_->data()+C.idx_, &ldc);
}

void MatrixView::symm(const char &tri, const char &side, const double &alpha, const MatrixView &symA, const double &beta, MatrixView &C) const{
#ifndef PKG_DEBUG_OFF
	if ( (Nrow_ == 0) || (Ncol_ == 0) ) {
		throw string("ERROR: one of the dimensions is zero");
	}
	if (symA.getNrows() != symA.getNcols()) {
		throw string("ERROR: symmetric matrix symA has to be square in symm()");
	}
	if (side == 'l') {
		if ((Nrow_ > INT_MAX) || (symA.getNcols() > INT_MAX)) {
		throw string("ERROR: at least one matrix dimension too big to safely convert to int in symm()");
		}
	} else if (side == 'r') {
		if ((symA.getNrows() > INT_MAX) || (Ncol_ > INT_MAX)) {
			throw string("ERROR: at least one matrix dimension too big to safely convert to int in symm()");
		}
	}

	if ((symA.getNcols() != Nrow_) && (side == 'l')) { // AB
		throw string("ERROR: Incompatible dimensions between B and A in symm()");
	}
	if ((symA.getNrows() != Ncol_) && (side == 'r')) { // BA
		throw string("ERROR: Incompatible dimensions between A and B in symm()");
	}
#endif

	int m;
	int n;
	if (side == 'l') { // AB
		m = static_cast<int>(symA.getNrows());
		n = static_cast<int>(Ncol_);
		if ((C.getNrows() != symA.getNrows()) || (C.getNcols() != Ncol_)) {
			throw string("ERROR: wrong C matrix dimensions in symm()");
		}
	} else if (side == 'r') { // BA
		m = static_cast<int>(Nrow_);
		n = static_cast<int>(symA.getNcols());
		if ((C.getNrows() != Nrow_) || (C.getNcols() != symA.getNcols())) {
			throw string("ERROR: wrong C matrix dimensions in symm()");
		}
	} else {
		throw string("ERROR: unknown side indicator in symm()");
	}

	// final integer parameters
	const int lda = static_cast<int>(symA.getNrows());
	const int ldb = static_cast<int>(Nrow_);
	const int ldc = m; // for clarity

	dsymm_(&side, &tri, &m, &n, &alpha, symA.data_->data()+symA.idx_, &lda, data_->data()+idx_, &ldb, &beta, C.data_->data()+C.idx_, &ldc);
}
void MatrixView::symm(const char &tri, const char &side, const double &alpha, const MatrixViewConst &symA, const double &beta, MatrixView &C) const{
#ifndef PKG_DEBUG_OFF
	if ( (Nrow_ == 0) || (Ncol_ == 0) ) {
		throw string("ERROR: one of the dimensions is zero");
	}
	if (symA.getNrows() != symA.getNcols()) {
		throw string("ERROR: symmetric matrix symA has to be square in symm()");
	}
	if (side == 'l') {
		if ((Nrow_ > INT_MAX) || (symA.getNcols() > INT_MAX)) {
		throw string("ERROR: at least one matrix dimension too big to safely convert to int in symm()");
		}
	} else if (side == 'r') {
		if ((symA.getNrows() > INT_MAX) || (Ncol_ > INT_MAX)) {
			throw string("ERROR: at least one matrix dimension too big to safely convert to int in symm()");
		}
	}

	if ((symA.getNcols() != Nrow_) && (side == 'l')) { // AB
		throw string("ERROR: Incompatible dimensions between B and A in symm()");
	}
	if ((symA.getNrows() != Ncol_) && (side == 'r')) { // BA
		throw string("ERROR: Incompatible dimensions between A and B in symm()");
	}
#endif

	int m;
	int n;
	if (side == 'l') { // AB
		m = static_cast<int>(symA.getNrows());
		n = static_cast<int>(Ncol_);
		if ((C.getNrows() != symA.getNrows()) || (C.getNcols() != Ncol_)) {
			throw string("ERROR: wrong C matrix dimensions in symm()");
		}
	} else if (side == 'r') { // BA
		m = static_cast<int>(Nrow_);
		n = static_cast<int>(symA.getNcols());
		if ((C.getNrows() != Nrow_) || (C.getNcols() != symA.getNcols())) {
			throw string("ERROR: wrong C matrix dimensions in symm()");
		}
	} else {
		throw string("ERROR: unknown side indicator in symm()");
	}

	// final integer parameters
	const int lda = static_cast<int>(symA.getNrows());
	const int ldb = static_cast<int>(Nrow_);
	const int ldc = m; // for clarity

	dsymm_(&side, &tri, &m, &n, &alpha, symA.data_->data()+symA.idx_, &lda, data_->data()+idx_, &ldb, &beta, C.data_->data()+C.idx_, &ldc);
}

void MatrixView::symc(const char &tri, const double &alpha, const MatrixView &X, const size_t &xCol, const double &beta, vector<double> &y) const{
#ifndef PKG_DEBUG_OFF
	if ( (Nrow_ == 0) || (Ncol_ == 0) ) {
		throw string("ERROR: one of the dimensions is zero");
	}
	if (Ncol_ != Nrow_) {
		throw string("ERROR: symmetric matrix (current object) has to be square in symc()");
	}
	if ((Ncol_ > INT_MAX) || (X.getNrows() > INT_MAX)) {
		throw string("ERROR: at least one matrix dimension too big to safely convert to int in symc()");
	}
	if (X.getNrows() != Ncol_) {
		throw string("ERROR: Incompatible dimensions between A and X in symc()");
	}
	if (xCol >= X.getNcols()) {
		throw string("ERROR: column index out of range for matrix X in symc()");
	}
#endif
	if (y.size() < Nrow_) {
		y.resize(Nrow_);
	}

	// BLAS routine constants
	const int n    = static_cast<int>(Nrow_);
	const int lda  = n;
	const int incx = 1;
	const int incy = 1;

	const double *xbeg = X.data_->data() + X.idx_ + xCol*(X.Nrow_); // offset to the column of interest

	dsymv_(&tri, &n, &alpha, data_->data()+idx_, &lda, xbeg, &incx, &beta, y.data(), &incy);
}

void MatrixView::symc(const char &tri, const double &alpha, const MatrixViewConst &X, const size_t &xCol, const double &beta, vector<double> &y) const{
#ifndef PKG_DEBUG_OFF
	if ( (Nrow_ == 0) || (Ncol_ == 0) ) {
		throw string("ERROR: one of the dimensions is zero");
	}
	if (Ncol_ != Nrow_) {
		throw string("ERROR: symmetric matrix (current object) has to be square in symc()");
	}
	if ((Ncol_ > INT_MAX) || (X.getNrows() > INT_MAX)) {
		throw string("ERROR: at least one matrix dimension too big to safely convert to int in symc()");
	}
	if (X.getNrows() != Ncol_) {
		throw string("ERROR: Incompatible dimensions between A and X in symc()");
	}
	if (xCol >= X.getNcols()) {
		throw string("ERROR: column index out of range for matrix X in symc()");
	}
#endif
	if (y.size() < Nrow_) {
		y.resize(Nrow_);
	}

	// BLAS routine constants
	const int n    = static_cast<int>(Nrow_);
	const int lda  = n;
	const int incx = 1;
	const int incy = 1;

	const double *xbeg = X.data_->data() + X.idx_ + xCol*(X.Nrow_); // offset to the column of interest

	dsymv_(&tri, &n, &alpha, data_->data()+idx_, &lda, xbeg, &incx, &beta, y.data(), &incy);
}

void MatrixView::trm(const char &tri, const char &side, const bool &transA, const bool &uDiag, const double &alpha, const MatrixView &trA){
#ifndef PKG_DEBUG_OFF
	if ( (Nrow_ == 0) || (Ncol_ == 0) ) {
		throw string("ERROR: one of the dimensions is zero");
	}
	if (trA.getNrows() != trA.getNcols()) {
		throw string("ERROR: triangular matrix trA has to be square in trm()");
	}
	if (side == 'l') {
		if ((Nrow_ > INT_MAX) || (trA.getNcols() > INT_MAX)) {
		throw string("ERROR: at least one matrix dimension too big to safely convert to int in trm()");
		}
	} else if (side == 'r') {
		if ((trA.getNrows() > INT_MAX) || (Ncol_ > INT_MAX)) {
			throw string("ERROR: at least one matrix dimension too big to safely convert to int in trm()");
		}
	}

	if ((trA.getNcols() != Nrow_) && (side == 'l')) { // AB
		throw string("ERROR: Incompatible dimensions between B and A in trm()");
	}
	if ((trA.getNrows() != Ncol_) && (side == 'r')) { // BA
		throw string("ERROR: Incompatible dimensions between A and B in trm()");
	}
	if ((side != 'l') && (side != 'r')) {
		throw string("ERROR: unknown side indicator in symm()");
	}
#endif

	int m = static_cast<int>(Nrow_);
	int n = static_cast<int>(Ncol_);

	// final integer parameters
	const int lda = static_cast<int>(trA.getNrows());
	const int ldb = static_cast<int>(Nrow_);
	char tAtok = (transA ? 't' : 'n');
	char Dtok  = (uDiag ? 'u' : 'n');

	dtrmm_(&side, &tri, &tAtok, &Dtok, &m, &n, &alpha, trA.data_->data()+trA.idx_, &lda, data_->data()+idx_, &ldb);
}

void MatrixView::trm(const char &tri, const char &side, const bool &transA, const bool &uDiag, const double &alpha, const MatrixViewConst &trA){
#ifndef PKG_DEBUG_OFF
	if ( (Nrow_ == 0) || (Ncol_ == 0) ) {
		throw string("ERROR: one of the dimensions is zero");
	}
	if (trA.getNrows() != trA.getNcols()) {
		throw string("ERROR: triangular matrix trA has to be square in trm()");
	}
	if (side == 'l') {
		if ((Nrow_ > INT_MAX) || (trA.getNcols() > INT_MAX)) {
		throw string("ERROR: at least one matrix dimension too big to safely convert to int in trm()");
		}
	} else if (side == 'r') {
		if ((trA.getNrows() > INT_MAX) || (Ncol_ > INT_MAX)) {
			throw string("ERROR: at least one matrix dimension too big to safely convert to int in trm()");
		}
	}

	if ((trA.getNcols() != Nrow_) && (side == 'l')) { // AB
		throw string("ERROR: Incompatible dimensions between B and A in trm()");
	}
	if ((trA.getNrows() != Ncol_) && (side == 'r')) { // BA
		throw string("ERROR: Incompatible dimensions between A and B in trm()");
	}
	if ((side != 'l') && (side != 'r')) {
		throw string("ERROR: unknown side indicator in symm()");
	}
#endif

	int m = static_cast<int>(Nrow_);
	int n = static_cast<int>(Ncol_);

	// final integer parameters
	const int lda = static_cast<int>(trA.getNrows());
	const int ldb = static_cast<int>(Nrow_);
	char tAtok = (transA ? 't' : 'n');
	char Dtok  = (uDiag ? 'u' : 'n');

	dtrmm_(&side, &tri, &tAtok, &Dtok, &m, &n, &alpha, trA.data_->data()+trA.idx_, &lda, data_->data()+idx_, &ldb);
}

void MatrixView::gemm(const bool &transA, const double &alpha, const MatrixView &A, const bool &transB, const double &beta, MatrixView &C) const{
#ifndef PKG_DEBUG_OFF
	if ( (Nrow_ == 0) || (Ncol_ == 0) ) {
		throw string("ERROR: one of the dimensions is zero");
	}
	if ((A.getNcols() > INT_MAX) || (A.getNrows() > INT_MAX)) {
		throw string("ERROR: at least one A matrix dimension too big to safely convert to int in gemm()");
	}

	if (transB) {
		if (Nrow_ > INT_MAX) {
			throw string("ERROR: at least one B matrix dimension too big to safely convert to int in gemm()");
		}
	} else {
		if (Ncol_ > INT_MAX) {
			throw string("ERROR: at least one B matrix dimension too big to safely convert to int in gemm()");
		}
	}
	if (transA) {
		if (transB && (A.getNrows() != Ncol_)) {
			throw string("ERROR: Incompatible dimensions between A^T and B^T in gemm()");
		} else if (!transB && (A.getNrows() != Nrow_)){
			throw string("ERROR: Incompatible dimensions between A^T and B in gemm()");
		}

	} else {
		if (transB && (A.getNcols() != Ncol_)) {
			throw string("ERROR: Incompatible dimensions between A and B^T in gemm()");
		} else if (!transB && (A.getNcols() != Nrow_)) {
			throw string("ERROR: Incompatible dimensions between A and B in gemm()");
		}
	}
#endif

	char tAtok;
	char tBtok;

	int m;
	int k;
	int n;
	if (transA) {
		tAtok = 't';
		m     = static_cast<int>(A.getNcols());
		k     = static_cast<int>(A.getNrows());
		if (transB) {
			tBtok = 't';
			n     = static_cast<int>(Nrow_);
			if ((C.getNrows() != A.getNcols()) || (C.getNcols() != Nrow_)) {
				throw string("ERROR: incompatible C matrix dimensions in gemm()");
			}
		} else {
			tBtok = 'n';
			n     = static_cast<int>(Ncol_);
			if ((C.getNrows() != A.getNcols()) || (C.getNcols() != Ncol_)) {
				throw string("ERROR: incompatible C matrix dimensions in gemm()");
			}
		}
	} else {
		tAtok = 'n';
		m     = static_cast<int>(A.getNrows());
		k     = static_cast<int>(A.getNcols());
		if (transB) {
			tBtok = 't';
			n     = static_cast<int>(Nrow_);
			if ((C.getNrows() != A.getNrows()) || (C.getNcols() != Nrow_)) {
				throw string("ERROR: incompatible C matrix dimensions in gemm()");
			}
		} else {
			tBtok = 'n';
			n     = static_cast<int>(Ncol_);
			if ((C.getNrows() != A.getNrows()) || (C.getNcols() != Ncol_)) {
				throw string("ERROR: incompatible C matrix dimensions in gemm()");
			}
		}
	}

	const int lda = (transA ? k : m);
	const int ldb = (transB ? n : k);
	const int ldc = m;

	dgemm_(&tAtok, &tBtok, &m, &n, &k, &alpha, A.data_->data()+A.idx_, &lda, data_->data()+idx_, &ldb, &beta, C.data_->data()+C.idx_, &ldc);
}

void MatrixView::gemm(const bool &transA, const double &alpha, const MatrixViewConst &A, const bool &transB, const double &beta, MatrixView &C) const{
#ifndef PKG_DEBUG_OFF
	if ( (Nrow_ == 0) || (Ncol_ == 0) ) {
		throw string("ERROR: one of the dimensions is zero");
	}
	if ((A.getNcols() > INT_MAX) || (A.getNrows() > INT_MAX)) {
		throw string("ERROR: at least one A matrix dimension too big to safely convert to int in gemm()");
	}

	if (transB) {
		if (Nrow_ > INT_MAX) {
			throw string("ERROR: at least one B matrix dimension too big to safely convert to int in gemm()");
		}
	} else {
		if (Ncol_ > INT_MAX) {
			throw string("ERROR: at least one B matrix dimension too big to safely convert to int in gemm()");
		}
	}
	if (transA) {
		if (transB && (A.getNrows() != Ncol_)) {
			throw string("ERROR: Incompatible dimensions between A^T and B^T in gemm()");
		} else if (!transB && (A.getNrows() != Nrow_)){
			throw string("ERROR: Incompatible dimensions between A^T and B in gemm()");
		}

	} else {
		if (transB && (A.getNcols() != Ncol_)) {
			throw string("ERROR: Incompatible dimensions between A and B^T in gemm()");
		} else if (!transB && (A.getNcols() != Nrow_)) {
			throw string("ERROR: Incompatible dimensions between A and B in gemm()");
		}
	}
#endif

	char tAtok;
	char tBtok;

	int m;
	int k;
	int n;
	if (transA) {
		tAtok = 't';
		m     = static_cast<int>(A.getNcols());
		k     = static_cast<int>(A.getNrows());
		if (transB) {
			tBtok = 't';
			n     = static_cast<int>(Nrow_);
			if ((C.getNrows() != A.getNcols()) || (C.getNcols() != Nrow_)) {
				throw string("ERROR: incompatible C matrix dimensions in gemm()");
			}
		} else {
			tBtok = 'n';
			n     = static_cast<int>(Ncol_);
			if ((C.getNrows() != A.getNcols()) || (C.getNcols() != Ncol_)) {
				throw string("ERROR: incompatible C matrix dimensions in gemm()");
			}
		}
	} else {
		tAtok = 'n';
		m     = static_cast<int>(A.getNrows());
		k     = static_cast<int>(A.getNcols());
		if (transB) {
			tBtok = 't';
			n     = static_cast<int>(Nrow_);
			if ((C.getNrows() != A.getNrows()) || (C.getNcols() != Nrow_)) {
				throw string("ERROR: incompatible C matrix dimensions in gemm()");
			}
		} else {
			tBtok = 'n';
			n     = static_cast<int>(Ncol_);
			if ((C.getNrows() != A.getNrows()) || (C.getNcols() != Ncol_)) {
				throw string("ERROR: incompatible C matrix dimensions in gemm()");
			}
		}
	}

	const int lda = (transA ? k : m);
	const int ldb = (transB ? n : k);
	const int ldc = m;

	dgemm_(&tAtok, &tBtok, &m, &n, &k, &alpha, A.data_->data()+A.idx_, &lda, data_->data()+idx_, &ldb, &beta, C.data_->data()+C.idx_, &ldc);
}
void MatrixView::gemc(const bool &trans, const double &alpha, const MatrixView &X, const size_t &xCol, const double &beta, vector<double> &y) const {
#ifndef PKG_DEBUG_OFF
	if ( (Nrow_ == 0) || (Ncol_ == 0) ) {
		throw string("ERROR: one of the dimensions is zero");
	}
	if (trans) {
		if ((Nrow_ > INT_MAX) || (X.getNrows() > INT_MAX)) {
			throw string("ERROR: at least one matrix dimension too big to safely convert to int in gemc()");
		}
		if (Nrow_ != X.getNrows()) {
			throw string("ERROR: Incompatible dimensions between A and X in gemc()");

		}
	} else {
		if ((Ncol_ > INT_MAX) || (X.getNrows() > INT_MAX)) {
			throw string("ERROR: at least one matrix dimension too big to safely convert to int in gemc()");
		}
		if (Ncol_ != X.getNrows()) {
			throw string("ERROR: Incompatible dimensions between A and X in gemc()");

		}
	}
	if (xCol >= X.getNcols()) {
		throw string("ERROR: column index out of range for matrix X in gemc()");
	}
#endif

	if (y.size() < Nrow_) {
		y.resize(Nrow_);
	}

	// Establish constants for DGEMV
	const char tTok = (trans ? 't' : 'n');

	const int m    = static_cast<int>(Nrow_);
	const int n    = static_cast<int>(Ncol_);
	const int lda  = m;
	const int incx = 1;
	const int incy = 1;

	const double *xbeg = X.data_->data() + X.idx_ + xCol*(X.Nrow_); // offset to the column of interest

	dgemv_(&tTok, &m, &n, &alpha, data_->data() + idx_, &lda, xbeg, &incx, &beta, y.data(), &incy);

}

MatrixView& MatrixView::operator+=(const double &scal){
	for (size_t iElm = 0; iElm < Ncol_*Nrow_; iElm++) {
		data_->data()[iElm+idx_] += scal;
	}

	return *this;
}

MatrixView& MatrixView::operator*=(const double &scal){
	for (size_t iElm = 0; iElm < Ncol_*Nrow_; iElm++) {
		data_->data()[iElm+idx_] *= scal;
	}

	return *this;
}

MatrixView& MatrixView::operator-=(const double &scal){
	for (size_t iElm = 0; iElm < Ncol_*Nrow_; iElm++) {
		data_->data()[iElm+idx_] -= scal;
	}

	return *this;
}

MatrixView& MatrixView::operator/=(const double &scal){
	for (size_t iElm = 0; iElm < Ncol_*Nrow_; iElm++) {
		data_->data()[iElm+idx_] /= scal;
	}

	return *this;
}

void MatrixView::rowExpand(const Index &ind, MatrixView &out) const{
#ifndef PKG_DEBUG_OFF
	if (ind.groupNumber() != Ncol_) {
		throw string("ERROR: Number of Index groups not equal to number of columns in rowExpand()");
	}
	if ( (Nrow_ == 0) || (Ncol_ == 0) ) {
		throw string("ERROR: one of the dimensions is zero");
	}
	if ((ind.size() != out.Ncol_) || (Nrow_ != out.Nrow_)) {
		throw string("ERROR: Index size not equal to output number of rows in rowExpand()");
	}
#endif

	for (size_t oldCol = 0; oldCol < ind.groupNumber(); oldCol++) {
		// going through all the rows of Z that correspond to the old column of M
		for (auto &f : ind[oldCol]) {
			// copying the column of M
			memcpy(out.data_->data() + out.idx_ + f*Nrow_, data_->data() + idx_ + oldCol*Nrow_, Nrow_*sizeof(double));
		}
	}

}
void MatrixView::rowSums(vector<double> &sums) const{
	if (sums.size() < Nrow_) {
		sums.resize(Nrow_);
	}
	for (size_t iRow = 0; iRow < Nrow_; iRow++) {
		sums[iRow] = 0.0; // in case something was in the vector passed to the function and resize did not erase it
		for (size_t jCol = 0; jCol < Ncol_; jCol++) {
			// not necessarily mumerically stable. Revisit later
			sums[iRow] += data_->data()[idx_ + Nrow_*jCol + iRow];
		}
	}
}
void MatrixView::rowSums(const Index &ind, MatrixView &out) const{
#ifndef PKG_DEBUG_OFF
	if (ind.size() != Ncol_) {
		throw string("ERROR: Factor length not the same as number of columns in calling matrix in rowSums()");
	}
	if ( (Nrow_ == 0) || (Ncol_ == 0) ) {
		throw string("ERROR: one of the dimensions is zero");
	}
	if ((ind.groupNumber() != out.Ncol_) || (Nrow_ != out.Ncol_)) {
		throw string("ERROR: Index group number does not equal output column number in rowSums()");
	}
#endif
	fill(out.data_->data()+out.idx_, out.data_->data() + out.idx_ + (out.Ncol_*out.Nrow_), 0.0);

	for (size_t newCol = 0; newCol < ind.groupNumber(); newCol++) {
		// going through all the Index elments that correspond to the new column of out
		for (auto &f : ind[newCol]) {
			// summing the row elements of the current matrix with the group of colums
			for (size_t iRow = 0; iRow < Nrow_; iRow++) {
				out.data_->data()[out.idx_ + ind.groupNumber()*newCol + iRow] += data_->data()[idx_ + Nrow_*f + iRow];
			}
		}
	}

}
void MatrixView::rowMeans(vector<double> &means) const{
	if (means.size() < Nrow_) {
		means.resize(Nrow_);
	}
	for (size_t iRow = 0; iRow < Nrow_; iRow++) {
		means[iRow] = 0.0; // in case something was in the vector passed to the function and resize did not erase it
		for (size_t jCol = 0; jCol < Ncol_; jCol++) {
			// numerically stable recursive mean calculation. GSL does it this way.
			means[iRow] += (data_->data()[idx_ + Nrow_*jCol + iRow] - means[iRow])/static_cast<double>(jCol + 1);
		}
	}
}
void MatrixView::rowMeans(const Index &ind, MatrixView &out) const{
#ifndef PKG_DEBUG_OFF
	if (ind.size() != Ncol_) {
		throw string("ERROR: Factor length not the same as number of columns in calling matrix in rowMeans()");
	}
	if ( (Nrow_ == 0) || (Ncol_ == 0) ) {
		throw string("ERROR: one of the dimensions is zero");
	}
	if ((ind.groupNumber() != out.Ncol_) || (Nrow_ != out.Ncol_)) {
		throw string("ERROR: Index group number does not equal output column number in rowMeans()");
	}
#endif
	fill(out.data_->data()+out.idx_, out.data_->data() + out.idx_ + (out.Ncol_*out.Nrow_), 0.0);

	for (size_t newCol = 0; newCol < ind.groupNumber(); newCol++) {
		// going through all the Index elments that correspond to the new column of out
		double denom = 1.0;
		for (auto &f : ind[newCol]) {
			// summing the row elements of the current matrix with the group of colums
			for (size_t iRow = 0; iRow < Nrow_; iRow++) {
				out.data_->data()[out.idx_ + ind.groupNumber()*newCol + iRow] +=
						(data_->data()[idx_ + Nrow_*f + iRow] - out.data_->data()[out.idx_ + ind.groupNumber()*newCol + iRow])/denom;
			}
			denom += 1.0;
		}
	}

}

void MatrixView::colExpand(const Index &ind, MatrixView &out) const {
#ifndef PKG_DEBUG_OFF
	if (ind.groupNumber() != Nrow_) {
		throw string("ERROR: incorrect number of Index groups in colExpand()");
	}
	if ( (Nrow_ == 0) || (Ncol_ == 0) ) {
		throw string("ERROR: one of the dimensions is zero");
	}
	if ((ind.size() != out.Nrow_) || (Ncol_ != out.Ncol_)) {
		throw string("ERROR: the output matrix has wrong dimensions in colExpand()");
	}
#endif

	for (size_t oldRow = 0; oldRow < ind.groupNumber(); oldRow++) {
		// going through all the rows of Z that correspond to the old row of M
		for (auto &f : ind[oldRow]) {
			// copying the row of M
			for (size_t jCol = 0; jCol < Ncol_; jCol++) {
				out.data_->data()[out.idx_ + ind.size()*jCol + f] = data_->data()[idx_ + Nrow_*jCol + oldRow];
			}
		}
	}

}
void MatrixView::colSums(vector<double> &sums) const{
	if (sums.size() < Ncol_) {
		sums.resize(Ncol_);
	}
	for (size_t jCol = 0; jCol < Ncol_; jCol++) {
		sums[jCol] = 0.0; // in case something was in the vector passed to the function and resize did not erase it
		for (size_t iRow = 0; iRow < Nrow_; iRow++) {
			// not necessarily mumerically stable. Revisit later
			sums[jCol] += data_->data()[idx_ + Nrow_*jCol + iRow];
		}
	}
}
void MatrixView::colSums(const Index &ind, MatrixView &out) const {
#ifndef PKG_DEBUG_OFF
	if (ind.size() != Nrow_) {
		throw string("ERROR: Wrong total length of Index in colSums()");
	}
	if ( (Nrow_ == 0) || (Ncol_ == 0) ) {
		throw string("ERROR: one of the dimensions is zero");
	}
	if ((ind.groupNumber() != out.Nrow_) || (Ncol_ != out.Ncol_)) {
		throw string("ERROR: incorrect Index group number in colSums()");
	}
#endif

	fill(out.data_->data()+out.idx_, out.data_->data() + out.idx_ + (out.Ncol_*out.Nrow_), 0.0);

	for (size_t newRow = 0; newRow < ind.groupNumber(); newRow++) {
		// going through all the rows of Z that correspond to the new row of M
		for (auto &f : ind[newRow]) {
			// summing the rows of M within the group defined by rows of Z
			for (size_t jCol = 0; jCol < Ncol_; jCol++) {
				out.data_->data()[out.idx_ + ind.groupNumber()*jCol + newRow] += data_->data()[idx_ + Nrow_*jCol + f];
			}
		}
	}

}
void MatrixView::colMeans(vector<double> &means) const{
	if (means.size() < Ncol_) {
		means.resize(Ncol_);
	}
	for (size_t jCol = 0; jCol < Ncol_; jCol++) {
		means[jCol] = 0.0; // in case something was in the vector passed to the function and resize did not erase it
		for (size_t iRow = 0; iRow < Nrow_; iRow++) {
			// numerically stable recursive mean calculation. GSL does it this way.
			means[jCol] += (data_->data()[idx_ + Nrow_*jCol + iRow] - means[jCol])/static_cast<double>(iRow + 1);
		}
	}
}
void MatrixView::colMeans(const Index &ind, MatrixView &out) const {
#ifndef PKG_DEBUG_OFF
	if (ind.size() != Nrow_) {
		throw string("ERROR: Wrong total length of Index in colMeans()");
	}
	if ( (Nrow_ == 0) || (Ncol_ == 0) ) {
		throw string("ERROR: one of the dimensions is zero");
	}
	if ((ind.groupNumber() != out.Nrow_) || (Ncol_ != out.Ncol_)) {
		throw string("ERROR: incorrect Index group number in colMeans()");
	}
#endif

	fill(out.data_->data()+out.idx_, out.data_->data() + out.idx_ + (out.Ncol_*out.Nrow_), 0.0);

	for (size_t newRow = 0; newRow < ind.groupNumber(); newRow++) {
		double denom = 1.0;
		for (auto &f : ind[newRow]) {
			for (size_t jCol = 0; jCol < Ncol_; jCol++) {
				out.data_->data()[out.idx_ + ind.groupNumber()*jCol + newRow] +=
					(data_->data()[idx_ + Nrow_*jCol + f] - out.data_->data()[out.idx_ + ind.groupNumber()*jCol + newRow])/denom;
			}
			denom += 1.0;
		}
	}

}

void MatrixView::rowMultiply(const vector<double> &scalars){
#ifndef PKG_DEBUG_OFF
	if (scalars.size() != Ncol_) {
		throw string("ERROR: Vector of scalars has wrong length in rowMultiply(vector)");
	}
#endif

	for (size_t iRow = 0; iRow < Nrow_; iRow++) {
		for (size_t jCol = 0; jCol < Ncol_; jCol++) {
			data_->data()[idx_ + Nrow_*jCol + iRow] *= scalars[jCol];
		}
	}
}
void MatrixView::rowMultiply(const double &scalar, const size_t &iRow){
#ifndef PKG_DEBUG_OFF
	if (iRow >= Nrow_) {
		throw string("ERROR: Row index out of bounds in rowMultiply(scalar)");
	}
#endif
	for (size_t jCol = 0; jCol < Ncol_; jCol++) {
		data_->data()[idx_ + Nrow_*jCol + iRow] *= scalar;
	}
}
void MatrixView::colMultiply(const vector<double> &scalars){
#ifndef PKG_DEBUG_OFF
	if (scalars.size() != Nrow_) {
		throw string("ERROR: Vector of scalars has wrong length in colMultiply(vector)");
	}
#endif

	for (size_t jCol = 0; jCol < Ncol_; jCol++) {
		for (size_t iRow = 0; iRow < Nrow_; iRow++) {
			data_->data()[idx_ + Nrow_*jCol + iRow] *= scalars[iRow];
		}
	}
}
void MatrixView::colMultiply(const double &scalar, const size_t &jCol){
#ifndef PKG_DEBUG_OFF
	if (jCol >= Ncol_) {
		throw string("ERROR: Column index out of bounds in colMultiply(scalar)");
	}
#endif
	for (size_t iRow = 0; iRow < Nrow_; iRow++) {
		data_->data()[idx_ + Nrow_*jCol + iRow] *= scalar;
	}
}
void MatrixView::rowDivide(const vector<double> &scalars){
#ifndef PKG_DEBUG_OFF
	if (scalars.size() != Ncol_) {
		throw string("ERROR: Vector of scalars has wrong length in rowDivide(vector)");
	}
#endif

	for (size_t iRow = 0; iRow < Nrow_; iRow++) {
		for (size_t jCol = 0; jCol < Ncol_; jCol++) {
			data_->data()[idx_ + Nrow_*jCol + iRow] /= scalars[jCol];
		}
	}
}
void MatrixView::rowDivide(const double &scalar, const size_t &iRow){
#ifndef PKG_DEBUG_OFF
	if (iRow >= Nrow_) {
		throw string("ERROR: Row index out of bounds in rowDivide(scalar)");
	}
#endif
	for (size_t jCol = 0; jCol < Ncol_; jCol++) {
		data_->data()[idx_ + Nrow_*jCol + iRow] /= scalar;
	}
}
void MatrixView::colDivide(const vector<double> &scalars){
#ifndef PKG_DEBUG_OFF
	if (scalars.size() != Nrow_) {
		throw string("ERROR: Vector of scalars has wrong length in colDivide(vector)");
	}
#endif

	for (size_t jCol = 0; jCol < Ncol_; jCol++) {
		for (size_t iRow = 0; iRow < Nrow_; iRow++) {
			data_->data()[idx_ + Nrow_*jCol + iRow] /= scalars[iRow];
		}
	}
}
void MatrixView::colDivide(const double &scalar, const size_t &jCol){
#ifndef PKG_DEBUG_OFF
	if (jCol >= Ncol_) {
		throw string("ERROR: Column index out of bounds in colDivide(scalar)");
	}
#endif
	for (size_t iRow = 0; iRow < Nrow_; iRow++) {
		data_->data()[idx_ + Nrow_*jCol + iRow] /= scalar;
	}
}
void MatrixView::rowAdd(const vector<double> &scalars){
#ifndef PKG_DEBUG_OFF
	if (scalars.size() != Ncol_) {
		throw string("ERROR: Vector of scalars has wrong length in rowAdd(vector)");
	}
#endif

	for (size_t iRow = 0; iRow < Nrow_; iRow++) {
		for (size_t jCol = 0; jCol < Ncol_; jCol++) {
			data_->data()[idx_ + Nrow_*jCol + iRow] += scalars[jCol];
		}
	}
}
void MatrixView::rowAdd(const double &scalar, const size_t &iRow){
#ifndef PKG_DEBUG_OFF
	if (iRow >= Nrow_) {
		throw string("ERROR: Row index out of bounds in rowAdd(scalar)");
	}
#endif
	for (size_t jCol = 0; jCol < Ncol_; jCol++) {
		data_->data()[idx_ + Nrow_*jCol + iRow] += scalar;
	}
}
void MatrixView::colAdd(const vector<double> &scalars){
#ifndef PKG_DEBUG_OFF
	if (scalars.size() != Nrow_) {
		throw string("ERROR: Vector of scalars has wrong length in colAdd(vector)");
	}
#endif

	for (size_t jCol = 0; jCol < Ncol_; jCol++) {
		for (size_t iRow = 0; iRow < Nrow_; iRow++) {
			data_->data()[idx_ + Nrow_*jCol + iRow] += scalars[iRow];
		}
	}
}
void MatrixView::colAdd(const double &scalar, const size_t &jCol){
#ifndef PKG_DEBUG_OFF
	if (jCol >= Ncol_) {
		throw string("ERROR: Column index out of bounds in colAdd(scalar)");
	}
#endif
	for (size_t iRow = 0; iRow < Nrow_; iRow++) {
		data_->data()[idx_ + Nrow_*jCol + iRow] += scalar;
	}
}
void MatrixView::rowSub(const vector<double> &scalars){
#ifndef PKG_DEBUG_OFF
	if (scalars.size() != Ncol_) {
		throw string("ERROR: Vector of scalars has wrong length in rowSub(vector)");
	}
#endif

	for (size_t iRow = 0; iRow < Nrow_; iRow++) {
		for (size_t jCol = 0; jCol < Ncol_; jCol++) {
			data_->data()[idx_ + Nrow_*jCol + iRow] -= scalars[jCol];
		}
	}
}
void MatrixView::rowSub(const double &scalar, const size_t &iRow){
#ifndef PKG_DEBUG_OFF
	if (iRow >= Nrow_) {
		throw string("ERROR: Row index out of bounds in rowSub(scalar)");
	}
#endif
	for (size_t jCol = 0; jCol < Ncol_; jCol++) {
		data_->data()[idx_ + Nrow_*jCol + iRow] -= scalar;
	}
}
void MatrixView::colSub(const vector<double> &scalars){
#ifndef PKG_DEBUG_OFF
	if (scalars.size() != Nrow_) {
		throw string("ERROR: Vector of scalars has wrong length in colSub(vector)");
	}
#endif

	for (size_t jCol = 0; jCol < Ncol_; jCol++) {
		for (size_t iRow = 0; iRow < Nrow_; iRow++) {
			data_->data()[idx_ + Nrow_*jCol + iRow] -= scalars[iRow];
		}
	}
}
void MatrixView::colSub(const double &scalar, const size_t &jCol){
#ifndef PKG_DEBUG_OFF
	if (jCol >= Ncol_) {
		throw string("ERROR: Column index out of bounds in colSub(scalar)");
	}
#endif
	for (size_t iRow = 0; iRow < Nrow_; iRow++) {
		data_->data()[idx_ + Nrow_*jCol + iRow] -= scalar;
	}
}

// MatrixViewConst methods
MatrixViewConst::MatrixViewConst(MatrixViewConst &&inMat){
	if (this != &inMat) {
		data_ = inMat.data_;
		idx_  = inMat.idx_;
		Nrow_ = inMat.Nrow_;
		Ncol_ = inMat.Ncol_;

		inMat.data_ = nullptr;
		inMat.idx_  = 0;
		inMat.Nrow_ = 0;
		inMat.Ncol_ = 0;
	}
}
MatrixViewConst::MatrixViewConst(MatrixView &&inMat) : data_{move(inMat.data_)}, idx_{inMat.idx_}, Nrow_{inMat.Nrow_}, Ncol_{inMat.Ncol_} {
	inMat.data_ = nullptr;
	inMat.idx_  = 0;
	inMat.Nrow_ = 0;
	inMat.Ncol_ = 0;
}

MatrixViewConst& MatrixViewConst::operator=(MatrixViewConst &&inMat){
	if (this != &inMat) {
		data_ = inMat.data_;
		idx_  = inMat.idx_;
		Nrow_ = inMat.Nrow_;
		Ncol_ = inMat.Ncol_;

		inMat.data_ = nullptr;
		inMat.idx_  = 0;
		inMat.Nrow_ = 0;
		inMat.Ncol_ = 0;
	}

	return *this;
}
MatrixViewConst& MatrixViewConst::operator=(MatrixView &&inMat){
	data_ = inMat.data_;
	idx_  = inMat.idx_;
	Nrow_ = inMat.Nrow_;
	Ncol_ = inMat.Ncol_;

	inMat.data_ = nullptr;
	inMat.idx_  = 0;
	inMat.Nrow_ = 0;
	inMat.Ncol_ = 0;

	return *this;
}

double MatrixViewConst::getElem(const size_t& iRow, const size_t &jCol) const{
#ifndef PKG_DEBUG_OFF
	if ((iRow >= Nrow_) || (jCol >= Ncol_)) {
		throw string("ERROR: element out of range in getElem()");
	}
#endif

	return data_->data()[idx_ + Nrow_*jCol + iRow];
}

void MatrixViewConst::chol(MatrixView &out) const {
#ifndef PKG_DEBUG_OFF
	if (Nrow_ != Ncol_) {
		throw string("ERROR: matrix has to be symmetric for Cholesky decomposition");
	}
	if ( (Nrow_ == 0) || (Ncol_ == 0) ) {
		throw string("ERROR: one of the dimensions is zero");
	}
#endif

	if ((Nrow_ != out.Nrow_) || (Ncol_ != out.Ncol_)) {
		throw string("ERROR: wrong dimensions in output matrix in copy Cholesky decomposition");
	}
	memcpy(out.data_->data() + out.idx_, data_->data() + idx_, (Nrow_ * Ncol_)*sizeof(double));

	int info = 0;
	char tri = 'L';

	int N = static_cast<int>(Nrow_); // conversion should be safe: Nrow_ magnitude checked during construction
	dpotrf_(&tri, &N, out.data_->data() + idx_, &N, &info);
	if (info < 0) {
		throw string("ERROR: illegal matrix element in copy Cholesky decomposition");
	} else if (info > 0) {
		throw string("ERROR: matrix is not positive definite in copy Cholesky decomposition");
	}

}

void MatrixViewConst::cholInv(MatrixView &out) const {
#ifndef PKG_DEBUG_OFF
	if (Nrow_ != Ncol_) {
		throw string("ERROR: matrix has to be square for Cholesky inversion");
	}
	if ( (Nrow_ == 0) || (Ncol_ == 0) ) {
		throw string("ERROR: one of the dimensions is zero");
	}
#endif

	if ((Nrow_ != out.Nrow_) || (Ncol_ != out.Ncol_)) {
		throw string("ERROR: wrong dimensions in output matrix in copy Cholesky inversion");
	}
	memcpy(out.data_->data()+out.idx_, data_->data()+idx_, (Nrow_ * Ncol_)*sizeof(double));

	int info = 0;
	char tri = 'L';

	int N = static_cast<int>(Nrow_); // safe to convert: Nrow_ checked at construction
	dpotri_(&tri, &N, out.data_->data() + idx_, &N, &info);
	if (info < 0) {
		throw string("ERROR: illegal matrix element in copy Cholesky inversion");
	} else if (info > 0) {
		throw string("ERROR: a diagonal element of the matrix is zero. Cannot complete copy Cholesky inversion");
	}
	for (size_t iRow = 0; iRow < Nrow_; iRow++) {
		for (size_t jCol = 0; jCol < iRow; jCol++) {
			(*out.data_)[out.idx_ + Nrow_*iRow + jCol] = (*out.data_)[out.idx_ + Nrow_*jCol + iRow];
		}
	}
}

void MatrixViewConst::svdSafe(MatrixView &U, vector<double> &s) const {
#ifndef PKG_DEBUG_OFF
	if ( (Nrow_ == 0) || (Ncol_ == 0) ) {
		throw string("ERROR: one of the dimensions is zero");
	}
	if ((Nrow_ != U.Nrow_) || (U.Nrow_ != U.Ncol_)) {
		throw string("ERROR: wrong dimensions of the U matrix in svdSafe()");
	}
#endif

	double *dataCopy = new double[Nrow_ * Ncol_];
	memcpy(dataCopy, data_->data()+idx_, (Nrow_ * Ncol_)*sizeof(double));

	if (s.size() < Ncol_) {
		s.resize(Ncol_, 0.0);
	}
	int Nvt = 1;
	vector<double>vt(1, 0.0);
	int resSVD = 0;
	int Nw = -1;    // set this to pre-run dgesvd_ for calculation of workspace size
	vector<double>workArr(1, 0.0);
	char jobu  = 'A';
	char jobvt = 'N';
	// the folloeing casts are safe because the dimensions are checked at construction
	int Nr = static_cast<int>(Nrow_);
	int Nc = static_cast<int>(Ncol_);

	// first calculate working space
	dgesvd_(&jobu, &jobvt, &Nr, &Nc, dataCopy, &Nr, s.data(), U.data_->data()+U.idx_, &Nr, vt.data(), &Nvt, workArr.data(), &Nw, &resSVD);
	Nw = workArr[0];
	workArr.resize(Nw, 0.0);
	dgesvd_(&jobu, &jobvt, &Nr, &Nc, dataCopy, &Nr, s.data(), U.data_->data()+U.idx_, &Nr, vt.data(), &Nvt, workArr.data(), &Nw, &resSVD);
	workArr.resize(0);
	if (resSVD < 0) {
		throw string("ERROR: illegal matrix element in safe SVD");
	} else if (resSVD > 0){
		throw string("ERROR: DBDSQR did not converge in safe SVD");
	}
	delete [] dataCopy;
}

void MatrixViewConst::eigenSafe(const char &tri, MatrixView &U, vector<double> &lam) const{
#ifndef PKG_DEBUG_OFF
	if (Nrow_ != Ncol_) {
		throw string("ERROR: matrix has to be at least square in eigen()");
	}
	if ( (Nrow_ == 0) || (Ncol_ == 0) ) {
		throw string("ERROR: one of the dimensions is zero");
	}
	if ((Ncol_ > U.Nrow_) || (Ncol_ > U.Ncol_)) {
		throw string("ERROR: wrong U matrix dimensions in eigenSafe()");
	}
#endif

	// test the output size and adjust if necessary
	if (Ncol_ > lam.size()) {
		lam.resize(Ncol_, 0.0);
	}

	char jobz  = 'V'; // computing eigenvectors
	char range = 'A'; // doing all of them
	char uplo;
	if (tri == 'u') {
		uplo = 'U';
	} else if (tri == 'l'){
		uplo = 'L';
	} else {
		throw string("ERROR: unknown triangle indicator in eigen()");
	}
	int N   = static_cast<int>(Nrow_);
	int lda = static_cast<int>(Nrow_);
	// placeholder variables. Not referenced since we are computing all eigenvectors
	double vl = 0.0;
	double vu = 0.0;
	int il = 0;
	int iu = 0;

	double abstol = sqrt(numeric_limits<double>::epsilon()); // absolute tolerance. Shouldn't be too close to epsilon since I don't need very precise estimation of small eigenvalues

	int M   = N;
	int ldz = N;

	vector<int> isuppz(2*M, 0);
	vector<double> work(1, 0.0);        // workspace; size will be determined
	int lwork = -1;          // to start; this lets us determine workspace size
	vector<int> iwork(1, 0); // integer workspace; size to be calculated
	int liwork = -1;         // to start; this lets us determine integer workspace size
	int info = 0;

	double *dataCopy = new double[Nrow_ * Ncol_];
	memcpy(dataCopy, data_->data()+idx_, (Nrow_ * Ncol_)*sizeof(double));

	dsyevr_(&jobz, &range, &uplo, &N, dataCopy, &lda, &vl, &vu, &il, &iu, &abstol, &M, lam.data(), U.data_->data()+U.idx_, &ldz, isuppz.data(), work.data(), &lwork, iwork.data(), &liwork, &info);

	lwork  = work[0];
	work.resize(static_cast<size_t>(lwork), 0.0);
	liwork = iwork[0];
	iwork.resize(static_cast<size_t>(liwork), 0);

	// run the actual estimation
	dsyevr_(&jobz, &range, &uplo, &N, dataCopy, &lda, &vl, &vu, &il, &iu, &abstol, &M, lam.data(), U.data_->data()+U.idx_, &ldz, isuppz.data(), work.data(), &lwork, iwork.data(), &liwork, &info);

	delete [] dataCopy;

	// set tiny eigenvalues to exactly zero
	for (auto &l : lam) {
		if (fabs(l) <= abstol) {
			l = 0.0;
		}
	}
}

void MatrixViewConst::eigenSafe(const char &tri, const size_t &n, MatrixView &U, vector<double> &lam) const{
#ifndef PKG_DEBUG_OFF
	if (Nrow_ != Ncol_) {
		throw string("ERROR: matrix has to be at least square in eigen()");
	}
	if (Nrow_ < n) {
		throw string("ERROR: the input number of eigenvalues greater than matrix dimensions");
	}
	if ( (Nrow_ == 0) || (Ncol_ == 0) ) {
		throw string("ERROR: one of the dimensions is zero");
	}
#endif

	if (Nrow_ == n) { // if we are doing all of them, just run regular eigen()
		MatrixViewConst::eigenSafe(tri, U, lam);
		return;
	}


	char jobz  = 'V'; // computing eigenvectors
	char range = 'I'; // doing some of them
	char uplo;
	if (tri == 'u') {
		uplo = 'U';
	} else if (tri == 'l'){
		uplo = 'L';
	} else {
		throw string("ERROR: unknown triangle indicator in eigen()");
	}
	int N   = static_cast<int>(Nrow_);
	int lda = static_cast<int>(Nrow_);
	// placeholder variables. Not referenced since we are computing a certain number of eigenvectors, not based on the values of the eigenvalues
	double vl = 0.0;
	double vu = 0.0;
	int il = N - static_cast<int>(n) + 1; // looks like the count base-1
	int iu = N; // do all the remaining eigenvalues

	double abstol = sqrt(numeric_limits<double>::epsilon()); // absolute tolerance. Shouldn't be too close to epsilon since I don't need very precise estimation of small eigenvalues

	int M   = iu - il + 1;
	int ldz = N;

	// test the output size and adjust if necessary
	if ((Nrow_ > U.Nrow_) || (static_cast<size_t>(M) > U.Ncol_)) {
		throw string("ERROR: wrong output matrix dimensions in eigenSafe()");
	}
	if (static_cast<size_t>(M) > lam.size()) {
		lam.resize(static_cast<size_t>(M), 0.0);
	}

	vector<int> isuppz(2*M, 0);
	vector<double> work(1, 0.0);        // workspace; size will be determined
	int lwork = -1;          // to start; this lets us determine workspace size
	vector<int> iwork(1, 0); // integer workspace; size to be calculated
	int liwork = -1;         // to start; this lets us determine integer workspace size
	int info = 0;

	double *dataCopy = new double[Nrow_ * Ncol_];
	memcpy(dataCopy, data_->data()+idx_, (Nrow_ * Ncol_)*sizeof(double));

	dsyevr_(&jobz, &range, &uplo, &N, dataCopy, &lda, &vl, &vu, &il, &iu, &abstol, &M, lam.data(), U.data_->data()+U.idx_, &ldz, isuppz.data(), work.data(), &lwork, iwork.data(), &liwork, &info);

	lwork  = work[0];
	work.resize(static_cast<size_t>(lwork), 0.0);
	liwork = iwork[0];
	iwork.resize(static_cast<size_t>(liwork), 0);

	// run the actual estimation
	dsyevr_(&jobz, &range, &uplo, &N, dataCopy, &lda, &vl, &vu, &il, &iu, &abstol, &M, lam.data(), U.data_->data()+U.idx_, &ldz, isuppz.data(), work.data(), &lwork, iwork.data(), &liwork, &info);

	delete [] dataCopy;

	// set tiny eigenvalues to exactly zero
	for (auto &l : lam) {
		if (fabs(l) <= abstol) {
			l = 0.0;
		}
	}
}

void MatrixViewConst::syrk(const char &tri, const double &alpha, const double &beta, MatrixView &C) const {
#ifndef PKG_DEBUG_OFF
	if ((Ncol_ > INT_MAX) || (Nrow_ > INT_MAX)) {
		throw string("ERROR: at least one matrix dimension too big to safely convert to int in syrk()");
	}
	if ( (Nrow_ == 0) || (Ncol_ == 0) ) {
		throw string("ERROR: one of the dimensions is zero");
	}
	if ((C.getNrows() != Ncol_) || (C.getNcols() != Ncol_)) {
		throw string("ERROR: wrong dimensions of the C matrix in syrk()");
	}
#endif

	// integer parameters
	const int n   = static_cast<int>(Ncol_);
	const int k   = static_cast<int>(Nrow_);
	const int lda = static_cast<int>(Nrow_);
	const int ldc = static_cast<int>(Ncol_);

	// transpose token
	const char trans = 't';

	dsyrk_(&tri, &trans, &n, &k, &alpha, data_->data()+idx_, &lda, &beta, C.data_->data()+C.idx_, &ldc);
}

void MatrixViewConst::tsyrk(const char &tri, const double &alpha, const double &beta, MatrixView &C) const {
#ifndef PKG_DEBUG_OFF
	if ((Ncol_ > INT_MAX) || (Nrow_ > INT_MAX)) {
		throw string("ERROR: at least one matrix dimension too big to safely convert to int in tsyrk()");
	}
	if ( (Nrow_ == 0) || (Ncol_ == 0) ) {
		throw string("ERROR: one of the dimensions is zero");
	}
	if ((C.getNrows() != Nrow_) || (C.getNcols() != Nrow_)) {
		throw string("ERROR: wrong C matrix dimensions in tsyrk()");
	}
#endif

	// integer parameters
	const int n   = static_cast<int>(Nrow_);
	const int k   = static_cast<int>(Ncol_);
	const int lda = static_cast<int>(Nrow_);
	const int ldc = static_cast<int>(Nrow_);

	// transpose token
	const char trans = 'n';

	dsyrk_(&tri, &trans, &n, &k, &alpha, data_->data()+idx_, &lda, &beta, C.data_->data()+C.idx_, &ldc);
}

void MatrixViewConst::symm(const char &tri, const char &side, const double &alpha, const MatrixView &symA, const double &beta, MatrixView &C) const{
#ifndef PKG_DEBUG_OFF
	if ( (Nrow_ == 0) || (Ncol_ == 0) ) {
		throw string("ERROR: one of the dimensions is zero");
	}
	if (symA.getNrows() != symA.getNcols()) {
		throw string("ERROR: symmetric matrix symA has to be square in symm()");
	}
	if (side == 'l') {
		if ((Nrow_ > INT_MAX) || (symA.getNcols() > INT_MAX)) {
		throw string("ERROR: at least one matrix dimension too big to safely convert to int in symm()");
		}
	} else if (side == 'r') {
		if ((symA.getNrows() > INT_MAX) || (Ncol_ > INT_MAX)) {
			throw string("ERROR: at least one matrix dimension too big to safely convert to int in symm()");
		}
	}

	if ((symA.getNcols() != Nrow_) && (side == 'l')) { // AB
		throw string("ERROR: Incompatible dimensions between B and A in symm()");
	}
	if ((symA.getNrows() != Ncol_) && (side == 'r')) { // BA
		throw string("ERROR: Incompatible dimensions between A and B in symm()");
	}
#endif

	int m;
	int n;
	if (side == 'l') { // AB
		m = static_cast<int>(symA.getNrows());
		n = static_cast<int>(Ncol_);
		if ((C.getNrows() != symA.getNrows()) || (C.getNcols() != Ncol_)) {
			throw string("ERROR: wrong C matrix dimensions in symm()");
		}
	} else if (side == 'r') { // BA
		m = static_cast<int>(Nrow_);
		n = static_cast<int>(symA.getNcols());
		if ((C.getNrows() != Nrow_) || (C.getNcols() != symA.getNcols())) {
			throw string("ERROR: wrong C matrix dimensions in symm()");
		}
	} else {
		throw string("ERROR: unknown side indicator in symm()");
	}

	// final integer parameters
	const int lda = static_cast<int>(symA.getNrows());
	const int ldb = static_cast<int>(Nrow_);
	const int ldc = m; // for clarity

	dsymm_(&side, &tri, &m, &n, &alpha, symA.data_->data()+symA.idx_, &lda, data_->data()+idx_, &ldb, &beta, C.data_->data()+C.idx_, &ldc);
}
void MatrixViewConst::symm(const char &tri, const char &side, const double &alpha, const MatrixViewConst &symA, const double &beta, MatrixView &C) const{
#ifndef PKG_DEBUG_OFF
	if ( (Nrow_ == 0) || (Ncol_ == 0) ) {
		throw string("ERROR: one of the dimensions is zero");
	}
	if (symA.getNrows() != symA.getNcols()) {
		throw string("ERROR: symmetric matrix symA has to be square in symm()");
	}
	if (side == 'l') {
		if ((Nrow_ > INT_MAX) || (symA.getNcols() > INT_MAX)) {
		throw string("ERROR: at least one matrix dimension too big to safely convert to int in symm()");
		}
	} else if (side == 'r') {
		if ((symA.getNrows() > INT_MAX) || (Ncol_ > INT_MAX)) {
			throw string("ERROR: at least one matrix dimension too big to safely convert to int in symm()");
		}
	}

	if ((symA.getNcols() != Nrow_) && (side == 'l')) { // AB
		throw string("ERROR: Incompatible dimensions between B and A in symm()");
	}
	if ((symA.getNrows() != Ncol_) && (side == 'r')) { // BA
		throw string("ERROR: Incompatible dimensions between A and B in symm()");
	}
#endif

	int m;
	int n;
	if (side == 'l') { // AB
		m = static_cast<int>(symA.getNrows());
		n = static_cast<int>(Ncol_);
		if ((C.getNrows() != symA.getNrows()) || (C.getNcols() != Ncol_)) {
			throw string("ERROR: wrong C matrix dimensions in symm()");
		}
	} else if (side == 'r') { // BA
		m = static_cast<int>(Nrow_);
		n = static_cast<int>(symA.getNcols());
		if ((C.getNrows() != Nrow_) || (C.getNcols() != symA.getNcols())) {
			throw string("ERROR: wrong C matrix dimensions in symm()");
		}
	} else {
		throw string("ERROR: unknown side indicator in symm()");
	}

	// final integer parameters
	const int lda = static_cast<int>(symA.getNrows());
	const int ldb = static_cast<int>(Nrow_);
	const int ldc = m; // for clarity

	dsymm_(&side, &tri, &m, &n, &alpha, symA.data_->data()+symA.idx_, &lda, data_->data()+idx_, &ldb, &beta, C.data_->data()+C.idx_, &ldc);
}

void MatrixViewConst::symc(const char &tri, const double &alpha, const MatrixView &X, const size_t &xCol, const double &beta, vector<double> &y) const{
#ifndef PKG_DEBUG_OFF
	if ( (Nrow_ == 0) || (Ncol_ == 0) ) {
		throw string("ERROR: one of the dimensions is zero");
	}
	if (Ncol_ != Nrow_) {
		throw string("ERROR: symmetric matrix (current object) has to be square in symc()");
	}
	if ((Ncol_ > INT_MAX) || (X.getNrows() > INT_MAX)) {
		throw string("ERROR: at least one matrix dimension too big to safely convert to int in symc()");
	}
	if (X.getNrows() != Ncol_) {
		throw string("ERROR: Incompatible dimensions between A and X in symc()");
	}
	if (xCol >= X.getNcols()) {
		throw string("ERROR: column index out of range for matrix X in symc()");
	}
#endif
	if (y.size() < Nrow_) {
		y.resize(Nrow_);
	}

	// BLAS routine constants
	const int n    = static_cast<int>(Nrow_);
	const int lda  = n;
	const int incx = 1;
	const int incy = 1;

	const double *xbeg = X.data_->data() + X.idx_ + xCol*(X.Nrow_); // offset to the column of interest

	dsymv_(&tri, &n, &alpha, data_->data()+idx_, &lda, xbeg, &incx, &beta, y.data(), &incy);
}

void MatrixViewConst::gemm(const bool &transA, const double &alpha, const MatrixView &A, const bool &transB, const double &beta, MatrixView &C) const{
#ifndef PKG_DEBUG_OFF
	if ( (Nrow_ == 0) || (Ncol_ == 0) ) {
		throw string("ERROR: one of the dimensions is zero");
	}
	if ((A.getNcols() > INT_MAX) || (A.getNrows() > INT_MAX)) {
		throw string("ERROR: at least one A matrix dimension too big to safely convert to int in gemm()");
	}

	if (transB) {
		if (Nrow_ > INT_MAX) {
			throw string("ERROR: at least one B matrix dimension too big to safely convert to int in gemm()");
		}
	} else {
		if (Ncol_ > INT_MAX) {
			throw string("ERROR: at least one B matrix dimension too big to safely convert to int in gemm()");
		}
	}
	if (transA) {
		if (transB && (A.getNrows() != Ncol_)) {
			throw string("ERROR: Incompatible dimensions between A^T and B^T in gemm()");
		} else if (!transB && (A.getNrows() != Nrow_)){
			throw string("ERROR: Incompatible dimensions between A^T and B in gemm()");
		}

	} else {
		if (transB && (A.getNcols() != Ncol_)) {
			throw string("ERROR: Incompatible dimensions between A and B^T in gemm()");
		} else if (!transB && (A.getNcols() != Nrow_)) {
			throw string("ERROR: Incompatible dimensions between A and B in gemm()");
		}
	}
#endif

	char tAtok;
	char tBtok;

	int m;
	int k;
	int n;
	if (transA) {
		tAtok = 't';
		m     = static_cast<int>(A.getNcols());
		k     = static_cast<int>(A.getNrows());
		if (transB) {
			tBtok = 't';
			n     = static_cast<int>(Nrow_);
			if ((C.getNrows() != A.getNcols()) || (C.getNcols() != Nrow_)) {
				throw string("ERROR: incompatible C matrix dimensions in gemm()");
			}
		} else {
			tBtok = 'n';
			n     = static_cast<int>(Ncol_);
			if ((C.getNrows() != A.getNcols()) || (C.getNcols() != Ncol_)) {
				throw string("ERROR: incompatible C matrix dimensions in gemm()");
			}
		}
	} else {
		tAtok = 'n';
		m     = static_cast<int>(A.getNrows());
		k     = static_cast<int>(A.getNcols());
		if (transB) {
			tBtok = 't';
			n     = static_cast<int>(Nrow_);
			if ((C.getNrows() != A.getNrows()) || (C.getNcols() != Nrow_)) {
				throw string("ERROR: incompatible C matrix dimensions in gemm()");
			}
		} else {
			tBtok = 'n';
			n     = static_cast<int>(Ncol_);
			if ((C.getNrows() != A.getNrows()) || (C.getNcols() != Ncol_)) {
				throw string("ERROR: incompatible C matrix dimensions in gemm()");
			}
		}
	}

	const int lda = (transA ? k : m);
	const int ldb = (transB ? n : k);
	const int ldc = m;

	dgemm_(&tAtok, &tBtok, &m, &n, &k, &alpha, A.data_->data()+A.idx_, &lda, data_->data()+idx_, &ldb, &beta, C.data_->data()+C.idx_, &ldc);
}
void MatrixViewConst::gemm(const bool &transA, const double &alpha, const MatrixViewConst &A, const bool &transB, const double &beta, MatrixView &C) const{
#ifndef PKG_DEBUG_OFF
	if ( (Nrow_ == 0) || (Ncol_ == 0) ) {
		throw string("ERROR: one of the dimensions is zero");
	}
	if ((A.getNcols() > INT_MAX) || (A.getNrows() > INT_MAX)) {
		throw string("ERROR: at least one A matrix dimension too big to safely convert to int in gemm()");
	}

	if (transB) {
		if (Nrow_ > INT_MAX) {
			throw string("ERROR: at least one B matrix dimension too big to safely convert to int in gemm()");
		}
	} else {
		if (Ncol_ > INT_MAX) {
			throw string("ERROR: at least one B matrix dimension too big to safely convert to int in gemm()");
		}
	}
	if (transA) {
		if (transB && (A.getNrows() != Ncol_)) {
			throw string("ERROR: Incompatible dimensions between A^T and B^T in gemm()");
		} else if (!transB && (A.getNrows() != Nrow_)){
			throw string("ERROR: Incompatible dimensions between A^T and B in gemm()");
		}

	} else {
		if (transB && (A.getNcols() != Ncol_)) {
			throw string("ERROR: Incompatible dimensions between A and B^T in gemm()");
		} else if (!transB && (A.getNcols() != Nrow_)) {
			throw string("ERROR: Incompatible dimensions between A and B in gemm()");
		}
	}
#endif

	char tAtok;
	char tBtok;

	int m;
	int k;
	int n;
	if (transA) {
		tAtok = 't';
		m     = static_cast<int>(A.getNcols());
		k     = static_cast<int>(A.getNrows());
		if (transB) {
			tBtok = 't';
			n     = static_cast<int>(Nrow_);
			if ((C.getNrows() != A.getNcols()) || (C.getNcols() != Nrow_)) {
				throw string("ERROR: incompatible C matrix dimensions in gemm()");
			}
		} else {
			tBtok = 'n';
			n     = static_cast<int>(Ncol_);
			if ((C.getNrows() != A.getNcols()) || (C.getNcols() != Ncol_)) {
				throw string("ERROR: incompatible C matrix dimensions in gemm()");
			}
		}
	} else {
		tAtok = 'n';
		m     = static_cast<int>(A.getNrows());
		k     = static_cast<int>(A.getNcols());
		if (transB) {
			tBtok = 't';
			n     = static_cast<int>(Nrow_);
			if ((C.getNrows() != A.getNrows()) || (C.getNcols() != Nrow_)) {
				throw string("ERROR: incompatible C matrix dimensions in gemm()");
			}
		} else {
			tBtok = 'n';
			n     = static_cast<int>(Ncol_);
			if ((C.getNrows() != A.getNrows()) || (C.getNcols() != Ncol_)) {
				throw string("ERROR: incompatible C matrix dimensions in gemm()");
			}
		}
	}

	const int lda = (transA ? k : m);
	const int ldb = (transB ? n : k);
	const int ldc = m;

	dgemm_(&tAtok, &tBtok, &m, &n, &k, &alpha, A.data_->data()+A.idx_, &lda, data_->data()+idx_, &ldb, &beta, C.data_->data()+C.idx_, &ldc);
}
void MatrixViewConst::gemc(const bool &trans, const double &alpha, const MatrixView &X, const size_t &xCol, const double &beta, vector<double> &y) const {
#ifndef PKG_DEBUG_OFF
	if ( (Nrow_ == 0) || (Ncol_ == 0) ) {
		throw string("ERROR: one of the dimensions is zero");
	}
	if (trans) {
		if ((Nrow_ > INT_MAX) || (X.getNrows() > INT_MAX)) {
			throw string("ERROR: at least one matrix dimension too big to safely convert to int in gemc()");
		}
		if (Nrow_ != X.getNrows()) {
			throw string("ERROR: Incompatible dimensions between A and X in gemc()");

		}
	} else {
		if ((Ncol_ > INT_MAX) || (X.getNrows() > INT_MAX)) {
			throw string("ERROR: at least one matrix dimension too big to safely convert to int in gemc()");
		}
		if (Ncol_ != X.getNrows()) {
			throw string("ERROR: Incompatible dimensions between A and X in gemc()");

		}
	}
	if (xCol >= X.getNcols()) {
		throw string("ERROR: column index out of range for matrix X in gemc()");
	}
#endif

	if (y.size() < Nrow_) {
		y.resize(Nrow_);
	}

	// Establish constants for DGEMV
	const char tTok = (trans ? 't' : 'n');

	const int m    = static_cast<int>(Nrow_);
	const int n    = static_cast<int>(Ncol_);
	const int lda  = m;
	const int incx = 1;
	const int incy = 1;

	const double *xbeg = X.data_->data() + X.idx_ + xCol*(X.Nrow_); // offset to the column of interest

	dgemv_(&tTok, &m, &n, &alpha, data_->data() + idx_, &lda, xbeg, &incx, &beta, y.data(), &incy);

}

void MatrixViewConst::rowExpand(const Index &ind, MatrixView &out) const{
#ifndef PKG_DEBUG_OFF
	if (ind.groupNumber() != Ncol_) {
		throw string("ERROR: Number of Index groups not equal to number of columns in rowExpand()");
	}
	if ( (Nrow_ == 0) || (Ncol_ == 0) ) {
		throw string("ERROR: one of the dimensions is zero");
	}
	if ((ind.size() != out.Ncol_) || (Nrow_ != out.Nrow_)) {
		throw string("ERROR: Index size not equal to output number of rows in rowExpand()");
	}
#endif

	for (size_t oldCol = 0; oldCol < ind.groupNumber(); oldCol++) {
		// going through all the rows of Z that correspond to the old column of M
		for (auto &f : ind[oldCol]) {
			// copying the column of M
			memcpy(out.data_->data() + out.idx_ + f*Nrow_, data_->data() + idx_ + oldCol*Nrow_, Nrow_*sizeof(double));
		}
	}

}
void MatrixViewConst::rowSums(vector<double> &sums) const{
	if (sums.size() < Nrow_) {
		sums.resize(Nrow_);
	}
	for (size_t iRow = 0; iRow < Nrow_; iRow++) {
		sums[iRow] = 0.0; // in case something was in the vector passed to the function and resize did not erase it
		for (size_t jCol = 0; jCol < Ncol_; jCol++) {
			// not necessarily mumerically stable. Revisit later
			sums[iRow] += data_->data()[idx_ + Nrow_*jCol + iRow];
		}
	}
}
void MatrixViewConst::rowSums(const Index &ind, MatrixView &out) const{
#ifndef PKG_DEBUG_OFF
	if (ind.size() != Ncol_) {
		throw string("ERROR: Factor length not the same as number of columns in calling matrix in rowSums()");
	}
	if ( (Nrow_ == 0) || (Ncol_ == 0) ) {
		throw string("ERROR: one of the dimensions is zero");
	}
	if ((ind.groupNumber() != out.Ncol_) || (Nrow_ != out.Ncol_)) {
		throw string("ERROR: Index group number does not equal output column number in rowSums()");
	}
#endif
	fill(out.data_->data()+out.idx_, out.data_->data() + out.idx_ + (out.Ncol_*out.Nrow_), 0.0);

	for (size_t newCol = 0; newCol < ind.groupNumber(); newCol++) {
		// going through all the Index elments that correspond to the new column of out
		for (auto &f : ind[newCol]) {
			// summing the row elements of the current matrix with the group of colums
			for (size_t iRow = 0; iRow < Nrow_; iRow++) {
				out.data_->data()[out.idx_ + ind.groupNumber()*newCol + iRow] += data_->data()[idx_ + Nrow_*f + iRow];
			}
		}
	}

}
void MatrixViewConst::rowMeans(vector<double> &means) const{
	if (means.size() < Nrow_) {
		means.resize(Nrow_);
	}
	for (size_t iRow = 0; iRow < Nrow_; iRow++) {
		means[iRow] = 0.0; // in case something was in the vector passed to the function and resize did not erase it
		for (size_t jCol = 0; jCol < Ncol_; jCol++) {
			// numerically stable recursive mean calculation. GSL does it this way.
			means[iRow] += (data_->data()[idx_ + Nrow_*jCol + iRow] - means[iRow])/static_cast<double>(jCol + 1);
		}
	}
}
void MatrixViewConst::rowMeans(const Index &ind, MatrixView &out) const{
#ifndef PKG_DEBUG_OFF
	if (ind.size() != Ncol_) {
		throw string("ERROR: Factor length not the same as number of columns in calling matrix in rowMeans()");
	}
	if ( (Nrow_ == 0) || (Ncol_ == 0) ) {
		throw string("ERROR: one of the dimensions is zero");
	}
	if ((ind.groupNumber() != out.Ncol_) || (Nrow_ != out.Ncol_)) {
		throw string("ERROR: Index group number does not equal output column number in rowMeans()");
	}
#endif
	fill(out.data_->data()+out.idx_, out.data_->data() + out.idx_ + (out.Ncol_*out.Nrow_), 0.0);

	for (size_t newCol = 0; newCol < ind.groupNumber(); newCol++) {
		// going through all the Index elments that correspond to the new column of out
		double denom = 1.0;
		for (auto &f : ind[newCol]) {
			// summing the row elements of the current matrix with the group of colums
			for (size_t iRow = 0; iRow < Nrow_; iRow++) {
				out.data_->data()[out.idx_ + ind.groupNumber()*newCol + iRow] +=
						(data_->data()[idx_ + Nrow_*f + iRow] - out.data_->data()[out.idx_ + ind.groupNumber()*newCol + iRow])/denom;
			}
			denom += 1.0;
		}
	}

}

void MatrixViewConst::colExpand(const Index &ind, MatrixView &out) const {
#ifndef PKG_DEBUG_OFF
	if (ind.groupNumber() != Nrow_) {
		throw string("ERROR: incorrect number of Index groups in colExpand()");
	}
	if ( (Nrow_ == 0) || (Ncol_ == 0) ) {
		throw string("ERROR: one of the dimensions is zero");
	}
	if ((ind.size() != out.Nrow_) || (Ncol_ != out.Ncol_)) {
		throw string("ERROR: the output matrix has wrong dimensions in colExpand()");
	}
#endif

	for (size_t oldRow = 0; oldRow < ind.groupNumber(); oldRow++) {
		// going through all the rows of Z that correspond to the old row of M
		for (auto &f : ind[oldRow]) {
			// copying the row of M
			for (size_t jCol = 0; jCol < Ncol_; jCol++) {
				out.data_->data()[out.idx_ + ind.size()*jCol + f] = data_->data()[idx_ + Nrow_*jCol + oldRow];
			}
		}
	}

}
void MatrixViewConst::colSums(vector<double> &sums) const{
	if (sums.size() < Ncol_) {
		sums.resize(Ncol_);
	}
	for (size_t jCol = 0; jCol < Ncol_; jCol++) {
		sums[jCol] = 0.0; // in case something was in the vector passed to the function and resize did not erase it
		for (size_t iRow = 0; iRow < Nrow_; iRow++) {
			// not necessarily mumerically stable. Revisit later
			sums[jCol] += data_->data()[idx_ + Nrow_*jCol + iRow];
		}
	}
}
void MatrixViewConst::colSums(const Index &ind, MatrixView &out) const {
#ifndef PKG_DEBUG_OFF
	if (ind.size() != Nrow_) {
		throw string("ERROR: Wrong total length of Index in colSums()");
	}
	if ( (Nrow_ == 0) || (Ncol_ == 0) ) {
		throw string("ERROR: one of the dimensions is zero");
	}
	if ((ind.groupNumber() != out.Nrow_) || (Ncol_ != out.Ncol_)) {
		throw string("ERROR: incorrect Index group number in colSums()");
	}
#endif

	fill(out.data_->data()+out.idx_, out.data_->data() + out.idx_ + (out.Ncol_*out.Nrow_), 0.0);

	for (size_t newRow = 0; newRow < ind.groupNumber(); newRow++) {
		// going through all the rows of Z that correspond to the new row of M
		for (auto &f : ind[newRow]) {
			// summing the rows of M within the group defined by rows of Z
			for (size_t jCol = 0; jCol < Ncol_; jCol++) {
				out.data_->data()[out.idx_ + ind.groupNumber()*jCol + newRow] += data_->data()[idx_ + Nrow_*jCol + f];
			}
		}
	}

}
void MatrixViewConst::colMeans(vector<double> &means) const{
	if (means.size() < Ncol_) {
		means.resize(Ncol_);
	}
	for (size_t jCol = 0; jCol < Ncol_; jCol++) {
		means[jCol] = 0.0; // in case something was in the vector passed to the function and resize did not erase it
		for (size_t iRow = 0; iRow < Nrow_; iRow++) {
			// numerically stable recursive mean calculation. GSL does it this way.
			means[jCol] += (data_->data()[idx_ + Nrow_*jCol + iRow] - means[jCol])/static_cast<double>(iRow + 1);
		}
	}
}
void MatrixViewConst::colMeans(const Index &ind, MatrixView &out) const {
#ifndef PKG_DEBUG_OFF
	if (ind.size() != Nrow_) {
		throw string("ERROR: Wrong total length of Index in colMeans()");
	}
	if ( (Nrow_ == 0) || (Ncol_ == 0) ) {
		throw string("ERROR: one of the dimensions is zero");
	}
	if ((ind.groupNumber() != out.Nrow_) || (Ncol_ != out.Ncol_)) {
		throw string("ERROR: incorrect Index group number in colMeans()");
	}
#endif

	fill(out.data_->data()+out.idx_, out.data_->data() + out.idx_ + (out.Ncol_*out.Nrow_), 0.0);

	for (size_t newRow = 0; newRow < ind.groupNumber(); newRow++) {
		double denom = 1.0;
		for (auto &f : ind[newRow]) {
			for (size_t jCol = 0; jCol < Ncol_; jCol++) {
				out.data_->data()[out.idx_ + ind.groupNumber()*jCol + newRow] +=
					(data_->data()[idx_ + Nrow_*jCol + f] - out.data_->data()[out.idx_ + ind.groupNumber()*jCol + newRow])/denom;
			}
			denom += 1.0;
		}
	}

}

