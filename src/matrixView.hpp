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
 * This is the project header file containing class definitions and interface documentation.
 *
 */

#ifndef matrixView_hpp
#define matrixView_hpp

#include <vector>
#include <string>

#include "index.hpp"

using std::vector;
using std::string;
using std::move;

namespace BayesicSpace {
	// Forward declarations
	class MatrixView;
	class MatrixViewConst;

	/** \brief Matrix view of a `vector`
	 *
	 * This matrix class creates points to a portion of a vector, presenting it as a matrix. Matrix operations can then be perfomed, possibly modifying the data pointed to.
	 * The idea is similar to GSL's `matrix_view`.
	 * The matrix is column-major to comply with LAPACK and BLAS routines.
	 * Columns and rows are base-0. Range checking is done unless the flag -DLMRG_CHECK_OFF is set at compile time.
	 *
	 *
	 */
	class MatrixView {
		friend class MatrixViewConst;

	public:
		/** \brief Default constructor
		 *
		 */
		MatrixView() : data_{nullptr}, idx_{0}, Nrow_{0}, Ncol_{0} {};
		/** \brief Constructor pointing to a C++ `vector`
		 *
		 * Points to a portion of the vector starting from the provided index.
		 *
		 * \param[in] inVec target vector
		 * \param[in] idx start index
		 * \param[in] nrow number of rows
		 * \param[in] ncol number of columns
		 */
		MatrixView(vector<double> *inVec, const size_t &idx, const size_t &nrow, const size_t &ncol);

		/** \brief Destructor
		 *
		 */
		~MatrixView(){ data_ = nullptr; };

		/** \brief Copy constructor (deleted) */
		MatrixView(const MatrixView &inMat) = delete;

		/** \brief Copy assignment operator (deleted) */
		MatrixView& operator=(const MatrixView &inMat) = delete;
		/** \brief Move constructor
		 *
		 * \param[in] inMat object to be moved
		 */
		MatrixView(MatrixView &&inMat);
		/** \brief Move assignment operator
		 *
		 * \param[in] inMat object to be moved
		 * \return MatrixView target object
		 *
		 */
		MatrixView& operator=(MatrixView &&inMat);

		/** \brief Access to number of rows
		 *
		 * \return size_t number of rows
		 */
		size_t getNrows() const{return Nrow_; };
		/** \brief Access to number of columns
		 *
		 * \return size_t number of columns
		 */
		size_t getNcols() const{return Ncol_; };
		/** \brief Access to an element
		 *
		 * \param[in] iRow row number
		 * \param[in] jCol column number
		 * \return double element value
		 */
		double getElem(const size_t &iRow, const size_t &jCol) const;
		/** \brief Set element to a value
		 *
		 * \param[in] iRow row number
		 * \param[in] jCol column number
		 * \param[in] input input value
		 */
		void setElem(const size_t &iRow, const size_t &jCol, const double &input);
		/** \brief Copy data from a vector to a column
		 *
		 * Copies data from a vector to a specified column. If the vector is too long, the first Nrow_ elements are used.
		 *
		 * \param[in] jCol column index (0 base)
		 * \param[in] data vector with data
		 *
		 */
		void setCol(const size_t &jCol, const vector<double> &data);
		/** \brief Add a scalar to an element
		 *
		 * \param[in] iRow row number
		 * \param[in] jCol column number
		 * \param[in] input value to add
		 */
		void addToElem(const size_t &iRow, const size_t &jCol, const double &input);
		/** \brief Subtract a scalar from an element
		 *
		 * \param[in] iRow row number
		 * \param[in] jCol column number
		 * \param[in] input value to subtract
		 */
		void subtractFromElem(const size_t &iRow, const size_t &jCol, const double &input);
		/** \brief Multiply an element by a scalar
		 *
		 * \param[in] iRow row number
		 * \param[in] jCol column number
		 * \param[in] input value to multiply by
		 */
		void multiplyElem(const size_t &iRow, const size_t &jCol, const double &input);
		/** \brief Divide an element by a scalar
		 *
		 * \param[in] iRow row number
		 * \param[in] jCol column number
		 * \param[in] input value to divide by
		 */
		void divideElem(const size_t &iRow, const size_t &jCol, const double &input);
		/** \brief Permute columns
		 *
		 * Re-order columns of the matrix according to the provided index vector.
		 *
		 * \param[in] idx index vector
		 */
		void permuteCols(const vector<size_t> &idx);
		/** \brief Permute rows
		 *
		 * Re-order rows of the matrix according to the provided index vector.
		 *
		 * \param[in] idx index vector
		 */
		void permuteRows(const vector<size_t> &idx);

		// Linear algebra functions

		/** \brief In-place Cholesky decomposition
		 *
		 * Performs the Cholesky decomposition and stores the resulting matrix in the lower triangle of the same object.
		 */
		void chol();
		/** \brief Copy Cholesky decomposition
		 *
		 * Performs the Cholesky decomposition and stores the result in the lower triangle of the provided MatrixView object. The original object is left untouched.
		 *
		 * \param[out] out object where the result is to be stored
		 */
		void chol(MatrixView &out) const;
		/** \brief In-place Cholesky inverse
		 *
		 * Computes the inverse of a Cholesky decomposition and stores the resulting matrix in the same object, resulting in a symmetric matrix. The object is assumed to be a Cholesky decomposition already.
		 */
		void cholInv();
		/** \brief Copy Cholesky inverse
		 *
		 * Computes the inverse of a Cholesky decomposition and stores the result in the provided MatrixView object, resulting in a symmetric matrix. The original object is left untouched. The object is assumed to be a Cholesky decomposition already.
		 *
		 * \param[out] out object where the result is to be stored
		 */
		void cholInv(MatrixView &out) const;
		/** \brief In-place pseudoinverse
		 *
		 * Computes a pseudoinverse of a symmetric square matrix using eigendecomposition (using the LAPACK _DSYEVR_ routine). The matrix is replaced with its inverse. Only the lower triangle of the input matrix is addressed.
		 *
		 */
		void pseudoInv();
		/** \brief In-place pseudoinverse with log-determinant
		 *
		 * Computes a pseudoinverse and its log-pseudodeterminant of a symmetric square matrix using eigendecomposition (using the LAPACK _DSYEVR_ routine). The matrix is replaced with its inverse. Only the lower triangle of the input matrix is addressed.
		 *
		 * \param[out] lDet log-pseudodeterminant of the inverted matrix
		 */
		void pseudoInv(double &lDet);
		/** \brief Copy pseudoinverse
		 *
		 * Computes a pseudoinverse of a symmetric square matrix using eigendecomposition (using the LAPACK _DSYEVR_ routine). The calling matrix is left intact and the result is copied to the output. Only the lower triangle of the input matrix is addressed.
		 *
		 * \param[out] out object where the result is to be stored
		 */
		void pseudoInv(MatrixView &out) const;
		/** \brief Copy pseudoinverse with log-determinant
		 *
		 * Computes a pseudoinverse and its log-pseudodeterminant of a symmetric square matrix using eigendecomposition (using the LAPACK _DSYEVR_ routine). The calling matrix is left intact and the result is copied to the output. Only the lower triangle of the input matrix is addressed.
		 *
		 * \param[out] out object where the result is to be stored
		 * \param[out] lDet log-pseudodeterminant of the inverted matrix
		 */
		void pseudoInv(MatrixView &out, double &lDet) const;
		/** \brief Perform SVD
		 *
		 * Performs SVD and stores the \f$U\f$ vectors in a MatrixView object and singular values in a C++ vector. For now, only does the _DGESVD_ from LAPACK with no \f$V^{T}\f$ matrix. The data in the object are destroyed.
		 *
		 * \param[out] U \f$U\f$ vector matrix
		 * \param[out] s singular value vector
		 */
		void svd(MatrixView &U, vector<double> &s);
		/** \brief Perform "safe" SVD
		 *
		 * Performs SVD and stores the \f$U\f$ vectors in a MatrixView object and singular values in a C++ vector. For now, only does the _DGESVD_ from LAPACK with no \f$V^{T}\f$ matrix. The data in the object are preserved, leading to some loss of efficiency compared to svd().
		 *
		 * \param[out] U \f$U\f$ vector matrix
		 * \param[out] s singular value vector
		 */
		void svdSafe(MatrixView &U, vector<double> &s) const;
		/** \brief All eigenvalues and vectors of a symmetric matrix
		 *
		 * Interface to the _DSYEVR_ LAPACK routine. This routine is recommended as the fastest (especially for smaller matrices) in LAPACK benchmarks. It is assumed that the current object is symmetric. It is only checked for being square.
		 * The data in the relevant triangle are destroyed.
		 *
		 * \param[in] tri triangle ID ('u' for upper or 'l' for lower)
		 * \param[out] U matrix of eigenvectors
		 * \param[out] lam vector of eigenvalues in ascending order
		 *
		 */
		void eigen(const char &tri, MatrixView &U, vector<double> &lam);
		/** \brief Some eigenvalues and vectors of a symmetric matrix
		 *
		 * Computes top _n_ eigenvalues and vectors of a symmetric matrix. Interface to the _DSYEVR_ LAPACK routine. This routine is recommended as the fastest (especially for smaller matrices) in LAPACK benchmarks. It is assumed that the current object is symmetric. It is only checked for being square.
		 * The data in the relevant triangle are destroyed.
		 *
		 * \param[in] tri triangle ID ('u' for upper or 'l' for lower)
		 * \param[in] n number of largest eigenvalues to compute
		 * \param[out] U matrix of eigenvectors
		 * \param[out] lam vector of eigenvalues in ascending order
		 *
		 */
		void eigen(const char &tri, const size_t &n, MatrixView &U, vector<double> &lam);
		/** \brief All eigenvalues and vectors of a symmetric matrix ("safe")
		 *
		 * Interface to the _DSYEVR_ LAPACK routine. This routine is recommended as the fastest (especially for smaller matrices) in LAPACK benchmarks. It is assumed that the current object is symmetric. It is only checked for being square.
		 * The data are preserved, leading to some loss of efficiency compared to eigen().
		 *
		 * \param[in] tri triangle ID ('u' for upper or 'l' for lower)
		 * \param[out] U matrix of eigenvectors
		 * \param[out] lam vector of eigenvalues in ascending order
		 *
		 */
		void eigenSafe(const char &tri, MatrixView &U, vector<double> &lam) const;
		/** \brief Some eigenvalues and vectors of a symmetric matrix ("safe")
		 *
		 * Computes the top _n_ eigenvectors and values of a symmetric matrix. Interface to the _DSYEVR_ LAPACK routine. This routine is recommended as the fastest (especially for smaller matrices) in LAPACK benchmarks. It is assumed that the current object is symmetric. It is only checked for being square.
		 * The data are preserved, leading to some loss of efficiency compared to eigen().
		 *
		 * \param[in] tri triangle ID ('u' for upper or 'l' for lower)
		 * \param[in] n number of largest eigenvalues to compute
		 * \param[out] U matrix of eigenvectors
		 * \param[out] lam vector of eigenvalues in ascending order
		 *
		 */
		void eigenSafe(const char &tri, const size_t &n, MatrixView &U, vector<double> &lam) const;

		// BLAS interface
		/** \brief Inner self crossproduct
		 *
		 * Interface for the BLAS _DSYRK_ routine. This function updates the given symmetric matrix \f$C\f$ with the operation
		 *
		 * \f$C \leftarrow \alpha A^{T}A + \beta C \f$
		 *
		 * The _char_ parameter governs which triangle of \f$C\f$ is used to store the result ('u' is upper and 'l' is lower). Only the specified triangle of _C_ is changed.
		 *
		 * \param[in] tri \f$C\f$ triangle ID
		 * \param[in] alpha the \f$\alpha\f$ parameter
		 * \param[in] beta the \f$\beta\f$ parameter
		 * \param[in,out] C the result \f$C\f$ matrix
		 */
		void syrk(const char &tri, const double &alpha, const double &beta, MatrixView &C) const;
		/** \brief Outer self crossproduct
		 *
		 * Interface for the BLAS _DSYRK_ routine. This function updates the given symmetric matrix \f$C\f$ with the operation
		 *
		 * \f$C \leftarrow \alpha AA^{T} + \beta C \f$
		 *
		 * The _char_ parameter governs which triangle of \f$C\f$ is used to store the result ('u' is upper and 'l' is lower). Only the specified triangle of _C_ is changed.
		 *
		 * \param[in] tri \f$C\f$ triangle ID
		 * \param[in] alpha the \f$\alpha\f$ parameter
		 * \param[in] beta the \f$\beta\f$ parameter
		 * \param[in,out] C the result \f$C\f$ matrix
		 */
		void tsyrk(const char &tri, const double &alpha, const double &beta, MatrixView &C) const;
		/** \brief Multiply by symmetric matrix
		 *
		 * Multiply the _MatrixView_ object by a symmetric matrix. The interface for the BLAS _DSYMM_ routine. Updates the input/output matrix \f$C\f$
		 *
		 * \f$C \leftarrow \alpha AB + \beta C \f$
		 *
		 * if _side_ is 'l' (left) and
		 *
		 * \f$C \leftarrow \alpha BA + \beta C \f$
		 *
		 * if _side_ is 'r' (right). The symmetric \f$A\f$ matrix is provided as input, the method is called from the \f$B\f$ matrix.
		 *
		 * \param[in] tri \f$A\f$ triangle ID ('u' for upper or 'l' for lower)
		 * \param[in] side multiplication side
		 * \param[in] alpha the \f$\alpha\f$ constant
		 * \param[in] symA symmetric matrix \f$A\f$
		 * \param[in] beta the \f$\beta\f$ constant
		 * \param[in,out] C the result \f$C\f$ matrix
		 *
		 */
		void symm(const char &tri, const char &side, const double &alpha, const MatrixView &symA, const double &beta, MatrixView &C) const;
		/** \brief Multiply by symmetric `MatrixViewConst`
		 *
		 * Multiply the _MatrixView_ object by a symmetric matrix. The interface for the BLAS _DSYMM_ routine. Updates the input/output matrix \f$C\f$
		 *
		 * \f$C \leftarrow \alpha AB + \beta C \f$
		 *
		 * if _side_ is 'l' (left) and
		 *
		 * \f$C \leftarrow \alpha BA + \beta C \f$
		 *
		 * if _side_ is 'r' (right). The symmetric \f$A\f$ matrix is provided as input, the method is called from the \f$B\f$ matrix.
		 *
		 * \param[in] tri \f$A\f$ triangle ID ('u' for upper or 'l' for lower)
		 * \param[in] side multiplication side
		 * \param[in] alpha the \f$\alpha\f$ constant
		 * \param[in] symA symmetric matrix \f$A\f$
		 * \param[in] beta the \f$\beta\f$ constant
		 * \param[in,out] C the result \f$C\f$ matrix
		 *
		 */
		void symm(const char &tri, const char &side, const double &alpha, const MatrixViewConst &symA, const double &beta, MatrixView &C) const;
		/** Multiply symmetric matrix by a column of another matrix
		 *
		 * Multiply the _MatrixView_ object, which is symmetric, by a specified column of a _MatrixView_. An interface for the BLAS _DSYMV_ routine. Updates the input vector \f$y\f$
		 *
		 * \f$y \leftarrow \alpha AX_{\cdot j} + \beta y  \f$
		 *
		 * If the output vector is too short it is resized, adding zero elements as needed. If it is too long, only the first Nrow(A) elements are modified.
		 *
		 * \param[in] tri \f$A\f$ (focal object) triangle ID ('u' for upper or 'l' for lower)
		 * \param[in] alpha the \f$\alpha\f$ constant
		 * \param[in] X matrix \f$X\f$ whose column will be used
		 * \param[in] xCol column of \f$X\f$ to be used (0 base)
		 * \param[in] beta the \f$\beta\f$ constant
		 * \param[in,out] y result vector
		 *
		 */
		void symc(const char &tri, const double &alpha, const MatrixView &X, const size_t &xCol, const double &beta, vector<double> &y) const;
		/** Multiply symmetric matrix by a column of another matrix
		 *
		 * Multiply the _MatrixView_ object, which is symmetric, by a specified column of a _MatrixViewConst_. An interface for the BLAS _DSYMV_ routine. Updates the input vector \f$y\f$
		 *
		 * \f$y \leftarrow \alpha AX_{\cdot j} + \beta y  \f$
		 *
		 * If the output vector is too short it is resized, adding zero elements as needed. If it is too long, only the first Nrow(A) elements are modified.
		 *
		 * \param[in] tri \f$A\f$ (focal object) triangle ID ('u' for upper or 'l' for lower)
		 * \param[in] alpha the \f$\alpha\f$ constant
		 * \param[in] X matrix \f$X\f$ whose column will be used
		 * \param[in] xCol column of \f$X\f$ to be used (0 base)
		 * \param[in] beta the \f$\beta\f$ constant
		 * \param[in,out] y result vector
		 *
		 */
		void symc(const char &tri, const double &alpha, const MatrixViewConst &X, const size_t &xCol, const double &beta, vector<double> &y) const;
		/** \brief Multiply by triangular matrix
		 *
		 * Multiply the _MatrixView_ object by a triangular matrix \f$A\f$. The interface for the BLAS _DTRMM_ routine. Updates current object \f$B\f$
		 *
		 * \f$B \leftarrow \alpha op(A) B\f$
		 *
		 * if _side_ is 'l' (left) and
		 *
		 * \f$B \leftarrow \alpha B op(A)\f$
		 *
		 * if _side_ is 'r' (right).
		 * \f$op(A)\f$ is \f$A^T\f$ or \f$A\f$ if _transA_ is true or false, respectively. The triangular \f$A\f$ matrix is provided as input, the method is called from the \f$B\f$ matrix. The current object is replaced by the transformed resulting matrix.
		 *
		 * \param[in] tri \f$A\f$ triangle ID ('u' for upper or 'l' for lower)
		 * \param[in] side multiplication side
		 * \param[in] transA whether matrix \f$A\f$ should be transposed
		 * \param[in] uDiag whether \f$A\f$ unit-diagonal or not
		 * \param[in] alpha the \f$\alpha\f$ constant
		 * \param[in] trA triangular matrix \f$A\f$
		 *
		 */
		void trm(const char &tri, const char &side, const bool &transA, const bool &uDiag, const double &alpha, const MatrixView &trA);
		/** \brief Multiply by triangular _MatrixViewConst_
		 *
		 * Multiply the _MatrixView_ object by a triangular matrix \f$A\f$. The interface for the BLAS _DTRMM_ routine. Updates current object \f$B\f$
		 *
		 * \f$B \leftarrow \alpha op(A) B\f$
		 *
		 * if _side_ is 'l' (left) and
		 *
		 * \f$B \leftarrow \alpha B op(A)\f$
		 *
		 * if _side_ is 'r' (right).
		 * \f$op(A)\f$ is \f$A^T\f$ or \f$A\f$ if _transA_ is true or false, respectively. The triangular \f$A\f$ matrix is provided as input, the method is called from the \f$B\f$ matrix. The current object is replaced by the transformed resulting matrix.
		 *
		 * \param[in] tri \f$A\f$ triangle ID ('u' for upper or 'l' for lower)
		 * \param[in] side multiplication side
		 * \param[in] transA whether matrix \f$A\f$ should be transposed
		 * \param[in] uDiag whether \f$A\f$ unit-diagonal or not
		 * \param[in] alpha the \f$\alpha\f$ constant
		 * \param[in] trA triangular matrix \f$A\f$
		 *
		 */
		void trm(const char &tri, const char &side, const bool &transA, const bool &uDiag, const double &alpha, const MatrixViewConst &trA);
		/** \brief General matrix multiplication
		 *
		 * Interface for the BLAS _DGEMM_ routine. Updates the input/output matrix \f$C\f$
		 *
		 * \f$ C \leftarrow \alpha op(A)op(B) + \beta C \f$
		 *
		 * where \f$op(A)\f$ is \f$A^T\f$ or \f$A\f$ if _transA_ is true or false, respectively, and similarly for \f$op(B)\f$. The method is called from \f$B\f$.
		 *
		 * \param[in] transA whether \f$A\f$ should be transposed
		 * \param[in] alpha the \f$\alpha\f$ constant
		 * \param[in] A matrix \f$A\f$
		 * \param[in] transB whether \f$B\f$ should be transposed
		 * \param[in] beta the \f$\beta\f$ constant
		 * \param[in,out] C the result \f$C\f$ matrix
		 *
		 */
		void gemm(const bool &transA, const double &alpha, const MatrixView &A, const bool &transB, const double &beta, MatrixView &C) const;
		/** \brief General matrix multiplication with `MatrixViewConst`
		 *
		 * Interface for the BLAS _DGEMM_ routine. Updates the input/output matrix \f$C\f$
		 *
		 * \f$ C \leftarrow \alpha op(A)op(B) + \beta C \f$
		 *
		 * where \f$op(A)\f$ is \f$A^T\f$ or \f$A\f$ if _transA_ is true or false, respectively, and similarly for \f$op(B)\f$. The method is called from \f$B\f$.
		 *
		 * \param[in] transA whether \f$A\f$ should be transposed
		 * \param[in] alpha the \f$\alpha\f$ constant
		 * \param[in] A matrix \f$A\f$
		 * \param[in] transB whether \f$B\f$ should be transposed
		 * \param[in] beta the \f$\beta\f$ constant
		 * \param[in,out] C the result \f$C\f$ matrix
		 *
		 */
		void gemm(const bool &transA, const double &alpha, const MatrixViewConst &A, const bool &transB, const double &beta, MatrixView &C) const;
		/** \brief Multiply a general matrix by a column of another matrix
		 *
		 * Multiply the _MatrixView_ object by a specified column of another matrix. An interface for the BLAS _DGEMV_ routine. Updates the input vector \f$y\f$
		 *
		 * \f$y \leftarrow \alpha AX_{\cdot j} + \beta y  \f$
		 *
		 * or
		 *
		 * \f$y \leftarrow \alpha A^{T}X_{\cdot j} + \beta y  \f$
		 *
		 * If the output vector is too short it is resized, adding zero elements as needed. If it is too long, only the first Nrow(A) elements are modified.
		 *
		 * \param[in] trans whether \f$A\f$ (focal object) should be transposed
		 * \param[in] alpha the \f$\alpha\f$ constant
		 * \param[in] X matrix \f$X\f$ whose column will be used
		 * \param[in] xCol column of \f$X\f$ to be used (0 base)
		 * \param[in] beta the \f$\beta\f$ constant
		 * \param[in,out] y result vector
		 *
		 */
		void gemc(const bool &trans, const double &alpha, const MatrixView &X, const size_t &xCol, const double &beta, vector<double> &y) const;

		// compound assignment operators
		/** \brief MatrixView-scalar compound addition
		 *
		 * \param[in] scal scalar
		 * \return MatrixView result
		 *
		 */
		MatrixView& operator+=(const double &scal);
		/** \brief MatrixView-scalar compound product
		 *
		 * \param[in] scal scalar
		 * \return MatrixView result
		 *
		 */
		MatrixView& operator*=(const double &scal);
		/** \brief MatrixView-scalar compound subtraction
		 *
		 * \param[in] scal scalar
		 * \return MatrixView result
		 *
		 */
		MatrixView& operator-=(const double &scal);
		/** \brief MatrixView-scalar compound division
		 *
		 * \param[in] scal scalar
		 * \return MatrixView result
		 *
		 */
		MatrixView& operator/=(const double &scal);

		// column- and row-wise operations
		/** \brief Expand rows according to the provided index
		 *
		 * Each row is expanded, creating more columns. The output matrix must be of the correct size.
		 *
		 * \param[in] ind `Index` with groups corresponding to existing rows
		 * \param[out] out output `MatrixView`
		 *
		 */
		void rowExpand(const Index &ind, MatrixView &out) const;
		/** \brief Row sums
		 *
		 * Sums row elements and stores them in the provided vector. If vector length is smaller than necessary, the vector is expanded. Otherwise, the first \f$N_{row}\f$ elements are used.
		 *
		 * \param[out] sums vector of sums
		 *
		 */
		void rowSums(vector<double> &sums) const;
		/** \brief Sum rows in groups
		 *
		 * The row elements are summed within each group of the `Index`. The output matrix must have the correct dimensions (\f$N_{col}\f$ the new matrix equal to the number of groups).
		 *
		 * \param[in] ind `Index` with elements corresponding to rows
		 * \param[out] out output `MatrixView`
		 *
		 */
		void rowSums(const Index &ind, MatrixView &out) const;
		/** \brief Row means
		 *
		 * Calculates means among row elements and stores them in the provided vector. If vector length is smaller than necessary, the vector is expanded. Otherwise, the first \f$N_{row}\f$ elements are used.
		 *
		 * \param[out] means vector of means
		 *
		 */
		void rowMeans(vector<double> &means) const;
		/** \brief Row means in groups
		 *
		 * Means among row elements are calculated within each group of the `Index`. The output matrix must have the correct dimensions (\f$N_{col}\f$ the new matrix equal to the number of groups).
		 *
		 * \param[in] ind `Index` with elements corresponding to rows
		 * \param[out] out output `MatrixView`
		 *
		 */
		void rowMeans(const Index &ind, MatrixView &out) const;

		/** \brief Column sums
		 *
		 * Calculates sums of column elements and stores them in the provided vector. If vector length is smaller than necessary, the vector is expanded. Otherwise, the first \f$N_{col}\f$ elements are used.
		 *
		 * \param[out] sums vector of sums
		 *
		 */
		void colSums(vector<double> &sums) const;
		/** \brief Sum columns in groups
		 *
		 * The column elements are summed within each group of the `Index`. The output matrix must have the correct dimensions.
		 *
		 * \param[in] ind `Index` with elements corresponding to columns
		 * \param[out] out output `MatrixView`
		 *
		 */
		void colSums(const Index &ind, MatrixView &out) const;
		/** \brief Expand columns accoring to the provided index
		 *
		 * Columns are expanded to make more rows. The output matrix must be of correct size.
		 *
		 * \param[in] ind `Index` with groups corresponding to columns
		 * \param[out] out output `MatrixView`
		 *
		 */
		void colExpand(const Index &ind, MatrixView &out) const;
		/** \brief Column means
		 *
		 * Calculates means among rows in each column and stores them in the provided vector. If vector length is smaller than necessary, the vector is expanded. Otherwise, the first \f$N_{row}\f$ elements are used.
		 *
		 * \param[out] means vector of means
		 *
		 */
		void colMeans(vector<double> &means) const;
		/** \brief Column means in groups
		 *
		 * Means among column elements are calculated within each group of the `Index`. The output matrix must have the correct dimensions.
		 *
		 * \param[in] ind `Index` with elements corresponding to columns
		 * \param[out] out output `MatrixView`
		 *
		 */
		void colMeans(const Index &ind, MatrixView &out) const;
		/** \brief Multiply rows by a vector
		 *
		 * Entry-wise multiplication of each row by the provided vector. The current object is modified.
		 *
		 * \param[in] scalars vector of scalars to use for multiplication
		 *
		 */
		void rowMultiply(const vector<double> &scalars);
		/** \brief Multiply a row by a scalar
		 *
		 * Entry-wise multiplication of a given row by the provided scalar. The current object is modified.
		 *
		 * \param[in] scalar scalar to use for multiplication
		 * \param[in] iRow row index
		 *
		 */
		void rowMultiply(const double &scalar, const size_t &iRow);
		/** \brief Multiply columns by a vector
		 *
		 * Entry-wise multiplication of each column by the provided vector. The current object is modified.
		 *
		 * \param[in] scalars vector of scalars to use for multiplication
		 *
		 */
		void colMultiply(const vector<double> &scalars);
		/** \brief Multiply a column by a scalar
		 *
		 * Entry-wise multiplication of a given column by the provided scalar. The current object is modified.
		 *
		 * \param[in] scalar scalar to use for multiplication
		 * \param[in] jCol column index
		 *
		 */
		void colMultiply(const double &scalar, const size_t &jCol);
		/** \brief Divide rows by a vector
		 *
		 * Entry-wise division of each row by the provided vector. The current object is modified.
		 *
		 * \param[in] scalars vector of scalars to use for division
		 *
		 */
		void rowDivide(const vector<double> &scalars);
		/** \brief Divide a row by a scalar
		 *
		 * Entry-wise division of a given row by the provided scalar. The current object is modified.
		 *
		 * \param[in] scalar scalar to use for division
		 * \param[in] iRow row index
		 *
		 */
		void rowDivide(const double &scalar, const size_t &iRow);
		/** \brief Divide columns by a vector
		 *
		 * Entry-wise division of each column by the provided vector. The current object is modified.
		 *
		 * \param[in] scalars vector of scalars to use for division
		 *
		 */
		void colDivide(const vector<double> &scalars);
		/** \brief Divide a column by a scalar
		 *
		 * Entry-wise division of a given column by the provided scalar. The current object is modified.
		 *
		 * \param[in] scalar scalar to use for division
		 * \param[in] jCol column index
		 *
		 */
		void colDivide(const double &scalar, const size_t &jCol);
		/** \brief Add a vector to rows
		 *
		 * Entry-wise addition of a vector to each row. The current object is modified.
		 *
		 * \param[in] scalars vector of scalars to use for addition
		 *
		 */
		void rowAdd(const vector<double> &scalars);
		/** \brief Add a scalar to a row
		 *
		 * Entry-wise addition of a scalar to the given row. The current object is modified.
		 *
		 * \param[in] scalar scalar to use for addition
		 * \param[in] iRow row index
		 *
		 */
		void rowAdd(const double &scalar, const size_t &iRow);
		/** \brief Add a vector to columns
		 *
		 * Entry-wise addition of a vector to each column. The current object is modified.
		 *
		 * \param[in] scalars vector of scalars to use for addition
		 *
		 */
		void colAdd(const vector<double> &scalars);
		/** \brief Add a scalar to a column
		 *
		 * Entry-wise addition of a scalar to the given column. The current object is modified.
		 *
		 * \param[in] scalar scalar to use for addition
		 * \param[in] jCol column index
		 *
		 */
		void colAdd(const double &scalar, const size_t &jCol);
		/** \brief Subtract a vector from rows
		 *
		 * Entry-wise subtraction of a vector from each row. The current object is modified.
		 *
		 * \param[in] scalars vector of scalars to use for subtraction
		 *
		 */
		void rowSub(const vector<double> &scalars);
		/** \brief Subtract a scalar from a row
		 *
		 * Entry-wise subtraction of a scalar from the given row. The current object is modified.
		 *
		 * \param[in] scalar scalar to use for subtraction
		 * \param[in] iRow row index
		 *
		 */
		void rowSub(const double &scalar, const size_t &iRow);
		/** \brief Subtract a vector from columns
		 *
		 * Entry-wise subtraction of a vector from each column. The current object is modified.
		 *
		 * \param[in] scalars vector of scalars to use for subtraction
		 *
		 */
		void colSub(const vector<double> &scalars);
		/** \brief Subtract a scalar from a column
		 *
		 * Entry-wise subtraction of a scalar from the given column. The current object is modified.
		 *
		 * \param[in] scalar scalar to use for subtraction
		 * \param[in] jCol column index
		 *
		 */
		void colSub(const double &scalar, const size_t &jCol);
	private:
		/** \brief Pointer to a vector */
		vector<double> *data_;
		/** \brief Start position of the matrix view */
		size_t idx_;
		/** \brief Number of rows */
		size_t Nrow_;
		/** \brief Number of columns */
		size_t Ncol_;
	};

	/** \brief A `const` version of `MatrixView`
	 *
	 * This matrix class creates points to a portion of a vector, presenting it as a matrix. Matrix operations can then be perfomed, but are aguaranteed not to modify the vector pointed to.
	 * The idea is similar to GSL's `matrix_view`.
	 * The matrix is column-major to comply with LAPACK and BLAS routines.
	 * Columns and rows are base-0. Range checking is done unless the flag -DLMRG_CHECK_OFF is set at compile time.
	 *
	 */
	class MatrixViewConst {
		friend class MatrixView;
	public:
		/** \brief Default constructor
		 *
		 */
		MatrixViewConst() : data_{nullptr}, idx_{0}, Nrow_{0}, Ncol_{0} {};
		/** \brief Constructor pointing to a C++ `vector`
		 *
		 * Points to a portion of the vector starting from the provided index.
		 *
		 * \param[in] inVec target vector
		 * \param[in] idx start index
		 * \param[in] nrow number of rows
		 * \param[in] ncol number of columns
		 */
		MatrixViewConst(const vector<double> *inVec, const size_t &idx, const size_t &nrow, const size_t &ncol) : data_{inVec}, idx_{idx}, Nrow_{nrow}, Ncol_{ncol} {};

		/** \brief Destructor
		 *
		 */
		~MatrixViewConst(){ data_ = nullptr; };

		/** \brief Copy constructor (deleted) */
		MatrixViewConst(const MatrixViewConst &inMat) = delete;

		/** \brief Copy assignment operator (deleted) */
		MatrixViewConst& operator=(const MatrixViewConst &inMat) = delete;
		/** \brief Move constructor
		 *
		 * \param[in] inMat object to be moved
		 */
		MatrixViewConst(MatrixViewConst &&inMat);
		/** \brief Move constructor from `MatrixView`
		 *
		 * \param[in] inMat object to be moved
		 */
		MatrixViewConst(MatrixView &&inMat);
		/** \brief Move assignment operator
		 *
		 * \param[in] inMat object to be moved
		 * \return MatrixViewConst target object
		 *
		 */
		MatrixViewConst& operator=(MatrixViewConst &&inMat);
		/** \brief Move assignment operator from `MatrixView`
		 *
		 * \param[in] inMat object to be moved
		 * \return MatrixViewConst target object
		 *
		 */
		MatrixViewConst& operator=(MatrixView &&inMat);

		/** \brief Access to number of rows
		 *
		 * \return size_t number of rows
		 */
		size_t getNrows() const{return Nrow_; };
		/** \brief Access to number of columns
		 *
		 * \return size_t number of columns
		 */
		size_t getNcols() const{return Ncol_; };
		/** \brief Access to an element
		 *
		 * \param[in] iRow row number
		 * \param[in] jCol column number
		 * \return double element value
		 */
		double getElem(const size_t &iRow, const size_t &jCol) const;

		/** \brief Copy Cholesky decomposition
		 *
		 * Performs the Cholesky decomposition and stores the result in the lower triangle of the provided MatrixView object. The original object is left untouched.
		 *
		 * \param[out] out object where the result is to be stored
		 */
		void chol(MatrixView &out) const;
		/** \brief Copy Cholesky inverse
		 *
		 * Computes the inverse of a Cholesky decomposition and stores the result in the provided MatrixView object, resulting in a symmetric matrix. The original object is left untouched. The object is assumed to be a Cholesky decomposition already.
		 *
		 * \param[out] out object where the result is to be stored
		 */
		void cholInv(MatrixView &out) const;
		/** \brief Copy pseudoinverse
		 *
		 * Computes a pseudoinverse of a symmetric square matrix using eigendecomposition (using the LAPACK _DSYEVR_ routine). The calling matrix is left intact and the result is copied to the output. Only the lower triangle of the input matrix is addressed.
		 *
		 * \param[out] out object where the result is to be stored
		 */
		void pseudoInv(MatrixView &out) const;
		/** \brief Copy pseudoinverse with log-determinant
		 *
		 * Computes a pseudoinverse and its log-pseudodeterminant of a symmetric square matrix using eigendecomposition (using the LAPACK _DSYEVR_ routine). The calling matrix is left intact and the result is copied to the output. Only the lower triangle of the input matrix is addressed.
		 *
		 * \param[out] out object where the result is to be stored
		 * \param[out] lDet log-pseudodeterminant of the inverted matrix
		 */
		void pseudoInv(MatrixView &out, double &lDet) const;
		/** \brief Perform "safe" SVD
		 *
		 * Performs SVD and stores the \f$U\f$ vectors in a MatrixView object and singular values in a C++ vector. For now, only does the _DGESVD_ from LAPACK with no \f$V^{T}\f$ matrix. The data in the object are preserved, leading to some loss of efficiency compared to svd().
		 *
		 * \param[out] U \f$U\f$ vector matrix
		 * \param[out] s singular value vector
		 */
		void svdSafe(MatrixView &U, vector<double> &s) const;
		/** \brief All eigenvalues and vectors of a symmetric matrix ("safe")
		 *
		 * Interface to the _DSYEVR_ LAPACK routine. This routine is recommended as the fastest (especially for smaller matrices) in LAPACK benchmarks. It is assumed that the current object is symmetric. It is only checked for being square.
		 * The data are preserved, leading to some loss of efficiency compared to eigen().
		 *
		 * \param[in] tri triangle ID ('u' for upper or 'l' for lower)
		 * \param[out] U matrix of eigenvectors
		 * \param[out] lam vector of eigenvalues in ascending order
		 *
		 */
		void eigenSafe(const char &tri, MatrixView &U, vector<double> &lam) const;
		/** \brief Some eigenvalues and vectors of a symmetric matrix ("safe")
		 *
		 * Computes the top _n_ eigenvectors and values of a symmetric matrix. Interface to the _DSYEVR_ LAPACK routine. This routine is recommended as the fastest (especially for smaller matrices) in LAPACK benchmarks. It is assumed that the current object is symmetric. It is only checked for being square.
		 * The data are preserved, leading to some loss of efficiency compared to eigen().
		 *
		 * \param[in] tri triangle ID ('u' for upper or 'l' for lower)
		 * \param[in] n number of largest eigenvalues to compute
		 * \param[out] U matrix of eigenvectors
		 * \param[out] lam vector of eigenvalues in ascending order
		 *
		 */
		void eigenSafe(const char &tri, const size_t &n, MatrixView &U, vector<double> &lam) const;

		// BLAS interface
		/** \brief Inner self crossproduct
		 *
		 * Interface for the BLAS _DSYRK_ routine. This function updates the given symmetric matrix \f$C\f$ with the operation
		 *
		 * \f$C \leftarrow \alpha A^{T}A + \beta C \f$
		 *
		 * The _char_ parameter governs which triangle of \f$C\f$ is used to store the result ('u' is upper and 'l' is lower). Only the specified triangle is changed.
		 *
		 * \param[in] tri \f$C\f$ triangle ID
		 * \param[in] alpha the \f$\alpha\f$ parameter
		 * \param[in] beta the \f$\beta\f$ parameter
		 * \param[in,out] C the result \f$C\f$ matrix
		 */
		void syrk(const char &tri, const double &alpha, const double &beta, MatrixView &C) const;
		/** \brief Outer self crossproduct
		 *
		 * Interface for the BLAS _DSYRK_ routine. This function updates the given symmetric matrix \f$C\f$ with the operation
		 *
		 * \f$C \leftarrow \alpha AA^{T} + \beta C \f$
		 *
		 * The _char_ parameter governs which triangle of \f$C\f$ is used to store the result ('u' is upper and 'l' is lower). Only the specified triangle is changed.
		 *
		 * \param[in] tri \f$C\f$ triangle ID
		 * \param[in] alpha the \f$\alpha\f$ parameter
		 * \param[in] beta the \f$\beta\f$ parameter
		 * \param[in,out] C the result \f$C\f$ matrix
		 */
		void tsyrk(const char &tri, const double &alpha, const double &beta, MatrixView &C) const;
		/** \brief Multiply by symmetric matrix
		 *
		 * Multiply the `MatrixViewConst` object by a symmetric matrix. The interface for the BLAS _DSYMM_ routine. Updates the input/output matrix \f$C\f$
		 *
		 * \f$C \leftarrow \alpha AB + \beta C \f$
		 *
		 * if _side_ is 'l' (left) and
		 *
		 * \f$C \leftarrow \alpha BA + \beta C \f$
		 *
		 * if _side_ is 'r' (right). The symmetric \f$A\f$ matrix is provided as input, the method is called from the \f$B\f$ matrix.
		 *
		 * \param[in] tri \f$A\f$ triangle ID ('u' for upper or 'l' for lower)
		 * \param[in] side multiplication side
		 * \param[in] alpha the \f$\alpha\f$ constant
		 * \param[in] symA symmetric matrix \f$A\f$
		 * \param[in] beta the \f$\beta\f$ constant
		 * \param[in,out] C the result \f$C\f$ matrix
		 *
		 */
		void symm(const char &tri, const char &side, const double &alpha, const MatrixView &symA, const double &beta, MatrixView &C) const;
		/** \brief Multiply by symmetric `MatrixViewConst`
		 *
		 * Multiply the `MatrixViewConst` object by a symmetric matrix. The interface for the BLAS _DSYMM_ routine. Updates the input/output matrix \f$C\f$
		 *
		 * \f$C \leftarrow \alpha AB + \beta C \f$
		 *
		 * if _side_ is 'l' (left) and
		 *
		 * \f$C \leftarrow \alpha BA + \beta C \f$
		 *
		 * if _side_ is 'r' (right). The symmetric \f$A\f$ matrix is provided as input, the method is called from the \f$B\f$ matrix.
		 *
		 * \param[in] tri \f$A\f$ triangle ID ('u' for upper or 'l' for lower)
		 * \param[in] side multiplication side
		 * \param[in] alpha the \f$\alpha\f$ constant
		 * \param[in] symA symmetric matrix \f$A\f$
		 * \param[in] beta the \f$\beta\f$ constant
		 * \param[in,out] C the result \f$C\f$ matrix
		 *
		 */
		void symm(const char &tri, const char &side, const double &alpha, const MatrixViewConst &symA, const double &beta, MatrixView &C) const;
		/** Multiply symmetric matrix by a column of another matrix
		 *
		 * Multiply the `MatrixViewConst` object, which is symmetric, by a specified column of another matrix. An interface for the BLAS _DSYMV_ routine. Updates the input vector \f$y\f$
		 *
		 * \f$y \leftarrow \alpha AX_{\cdot j} + \beta y  \f$
		 *
		 * If the output vector is too short it is resized, adding zero elements as needed. If it is too long, only the first Nrow(A) elements are modified.
		 *
		 * \param[in] tri \f$A\f$ (focal object) triangle ID ('u' for upper or 'l' for lower)
		 * \param[in] alpha the \f$\alpha\f$ constant
		 * \param[in] X matrix \f$X\f$ whose column will be used
		 * \param[in] xCol column of \f$X\f$ to be used (0 base)
		 * \param[in] beta the \f$\beta\f$ constant
		 * \param[in,out] y result vector
		 *
		 */
		void symc(const char &tri, const double &alpha, const MatrixView &X, const size_t &xCol, const double &beta, vector<double> &y) const;
		/** \brief General matrix multiplication
		 *
		 * Interface for the BLAS _DGEMM_ routine. Updates the input/output matrix \f$C\f$
		 *
		 * \f$ C \leftarrow \alpha op(A)op(B) + \beta C \f$
		 *
		 * where \f$op(A)\f$ is \f$A^T\f$ or \f$A\f$ if _transA_ is true or false, respectively, and similarly for \f$op(B)\f$. The method is called from \f$B\f$.
		 *
		 * \param[in] transA whether \f$A\f$ should be transposed
		 * \param[in] alpha the \f$\alpha\f$ constant
		 * \param[in] A matrix \f$A\f$
		 * \param[in] transB whether \f$B\f$ should be transposed
		 * \param[in] beta the \f$\beta\f$ constant
		 * \param[in,out] C the result \f$C\f$ matrix
		 *
		 */
		void gemm(const bool &transA, const double &alpha, const MatrixView &A, const bool &transB, const double &beta, MatrixView &C) const;
		/** \brief General matrix multiplication with `MatrixViewConst`
		 *
		 * Interface for the BLAS _DGEMM_ routine. Updates the input/output matrix \f$C\f$
		 *
		 * \f$ C \leftarrow \alpha op(A)op(B) + \beta C \f$
		 *
		 * where \f$op(A)\f$ is \f$A^T\f$ or \f$A\f$ if _transA_ is true or false, respectively, and similarly for \f$op(B)\f$. The method is called from \f$B\f$.
		 *
		 * \param[in] transA whether \f$A\f$ should be transposed
		 * \param[in] alpha the \f$\alpha\f$ constant
		 * \param[in] A matrix \f$A\f$
		 * \param[in] transB whether \f$B\f$ should be transposed
		 * \param[in] beta the \f$\beta\f$ constant
		 * \param[in,out] C the result \f$C\f$ matrix
		 *
		 */
		void gemm(const bool &transA, const double &alpha, const MatrixViewConst &A, const bool &transB, const double &beta, MatrixView &C) const;
		/** \brief Multiply a general matrix by a column of another matrix
		 *
		 * Multiply the `MatrixViewConst` object by a specified column of another matrix. An interface for the BLAS _DGEMV_ routine. Updates the input vector \f$y\f$
		 *
		 * \f$y \leftarrow \alpha AX_{\cdot j} + \beta y  \f$
		 *
		 * or
		 *
		 * \f$y \leftarrow \alpha A^{T}X_{\cdot j} + \beta y  \f$
		 *
		 * If the output vector is too short it is resized, adding zero elements as needed. If it is too long, only the first Nrow(A) elements are modified.
		 *
		 * \param[in] trans whether \f$A\f$ (focal object) should be transposed
		 * \param[in] alpha the \f$\alpha\f$ constant
		 * \param[in] X matrix \f$X\f$ whose column will be used
		 * \param[in] xCol column of \f$X\f$ to be used (0 base)
		 * \param[in] beta the \f$\beta\f$ constant
		 * \param[in,out] y result vector
		 *
		 */
		void gemc(const bool &trans, const double &alpha, const MatrixView &X, const size_t &xCol, const double &beta, vector<double> &y) const;

		// column- and row-wise operations
		/** \brief Expand rows according to the provided index
		 *
		 * Each row is expanded, creating more columns. The output matrix must be of the correct size.
		 *
		 * \param[in] ind `Index` with groups corresponding to existing rows
		 * \param[out] out output `MatrixView`
		 *
		 */
		void rowExpand(const Index &ind, MatrixView &out) const;
		/** \brief Sum rows in groups
		 *
		 * The row elements are summed within each group of the `Index`. The output matrix must have the correct dimensions (\f$N_{col}\f$ the new matrix equal to the number of groups).
		 *
		 * \param[in] ind `Index` with elements corresponding to rows
		 * \param[out] out output `MatrixView`
		 *
		 */
		void rowSums(const Index &ind, MatrixView &out) const;
		/** \brief Row sums
		 *
		 * Sums row elements and stores them in the provided vector. If vector length is smaller than necessary, the vector is expanded. Otherwise, the first \f$N_{row}\f$ elements are used.
		 *
		 * \param[out] sums vector of sums
		 *
		 */
		void rowSums(vector<double> &sums) const;
		/** \brief Row means
		 *
		 * Calculates means among row elements and stores them in the provided vector. If vector length is smaller than necessary, the vector is expanded. Otherwise, the first \f$N_{row}\f$ elements are used.
		 *
		 * \param[out] means vector of means
		 *
		 */
		void rowMeans(vector<double> &means) const;
		/** \brief Row means in groups
		 *
		 * Means among row elements are calculated within each group of the `Index`. The output matrix must have the correct dimensions.
		 *
		 * \param[in] ind `Index` with elements corresponding to rows
		 * \param[out] out output `MatrixView`
		 *
		 */
		void rowMeans(const Index &ind, MatrixView &out) const;

		/** \brief Column sums
		 *
		 * Calculates sums of column elements and stores them in the provided vector. If vector length is smaller than necessary, the vector is expanded. Otherwise, the first \f$N_{col}\f$ elements are used.
		 *
		 * \param[out] sums vector of sums
		 *
		 */
		void colSums(vector<double> &sums) const;
		/** \brief Sum columns in groups
		 *
		 * The column elements are summed within each group of the `Index`. The output matrix must have the correct dimensions.
		 *
		 * \param[in] ind `Index` with elements corresponding to columns
		 * \param[out] out output `MatrixView`
		 *
		 */
		void colSums(const Index &ind, MatrixView &out) const;
		/** \brief Expand columns accoring to the provided index
		 *
		 * Columns are expanded to make more rows. The output matrix must be of correct size.
		 *
		 * \param[in] ind `Index` with groups corresponding to columns
		 * \param[out] out output `MatrixView`
		 *
		 */
		void colExpand(const Index &ind, MatrixView &out) const;
		/** \brief Column means
		 *
		 * Calculates means among rows in each column and stores them in the provided vector. If vector length is smaller than necessary, the vector is expanded. Otherwise, the first \f$N_{col}\f$ elements are used.
		 *
		 * \param[out] means vector of means
		 *
		 */
		void colMeans(vector<double> &means) const;
		/** \brief Column means in groups
		 *
		 * Means among column elements are calculated within each group of the `Index`. The output matrix must have the correct dimensions.
		 *
		 * \param[in] ind `Index` with elements corresponding to columns
		 * \param[out] out output `MatrixView`
		 *
		 */
		void colMeans(const Index &ind, MatrixView &out) const;
	private:
		/** \brief Pointer to a vector */
		const vector<double> *data_;
		/** \brief Start position of the matrix view */
		size_t idx_;
		/** \brief Number of rows */
		size_t Nrow_;
		/** \brief Number of columns */
		size_t Ncol_;
	};
}


#endif /* matrixView_hpp */
