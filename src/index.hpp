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

/// Connect lines with groups
/** \file
 * \author Anthony J. Greenberg
 * \copyright Copyright (c) 2017 Anthony J. Greenberg
 * \version 1.0
 *
 * Definitions and interface documentation for a class that relates individuals to groups, similar to an factor in R.
 *
 */

#ifndef index_hpp
#define index_hpp

#include <vector>
#include <string>

using std::vector;
using std::string;

namespace BayesicSpace {
	/** \brief Group index
	 *
	 * For each group, contains indexes of the lines that belong to it. Can also identify the group a given element belongs to. Group numbers need not be consecutive. Although group IDs are assumed to be base-0, everything should work even if they are not.
	 */
	class Index {

	public:
		/** \brief Default constructor */
		Index() {};
		/** \brief Group constructor
		 *
		 * Sets up empty groups.
		 *
		 * \param[in] Ngroups number of groups to set up
		 */
		Index(const size_t &Ngroups);
		/** \brief Array constructor
		 *
		 * The input array has an element for each line, and the value of that element is the base-0 group ID (i.e., if line _n_ is in the first group, then `arr[n] == 0`).
		 *
		 * \param[in] arr array of group IDs
		 * \param[in] N array length
		 */
		Index(const size_t *arr, const size_t &N);
		/** \brief Vector constructor
		 *
		 * The input vector has an element for each line, and the value of that element is the base-0 group ID (i.e., if line _n_ is in the first group, then `vec[n] == 0`).
		 *
		 * \param[in] vec array of group IDs
		 */
		Index(const vector<size_t> &vec);
		/** \brief File read constructor
		 *
		 * The input file has an entry for each line (separated by white space), and the value of that entry is the base-0 group ID.
		 * If the file cannot be opened, throws "Cannot open file file_name". If a negative group value is detected, thorws "Negative group ID".
		 *
		 * \param[in] inFileName input file name
		 */
		Index(const string &inFileName);
		/** \brief Copy constructor
		 *
		 * \param[in] in `Index` to be copied
		 * \return `Index` object
		 */
		Index(const Index &in) : index_{in.index_}, groupVal_{in.groupVal_} {};
		/** \brief Copy assignment operator
		 *
		 * \param[in] in object to be copied
		 * \return an `Index` object
		 */
		Index &operator=(const Index &in);
		/** \brief Move constructor
		 *
		 * \param[in] in `Index` object to be moved
		 * \return `Index` object
		 */
		Index(Index &&in) : index_{move(in.index_)}, groupVal_{move(in.groupVal_)} {};
		/** \brief Move assignment operator
		 *
		 * \param[in] in object to be moved
		 * \return an `Index` object
		 */
		Index &operator=(Index &&in);
		/** \brief Destructor */
		~Index(){};

		/** \brief Vector subscript operator
		 *
		 * Returns the index of group _i_.
		 *
		 * \param[in] i group index
		 * \return index of line IDs
		 */
		const vector<size_t> & operator[] (const size_t &i) const { return index_[i]; };

		/** \brief Group size
		 *
		 * \param[in] i group index
		 * \return size of the _i_th group
		 */
		size_t groupSize(const size_t &i) const {return index_[i].size(); };

		/** \brief Total sample size
		 *
		 * \return total sample size
		 */
		size_t size() const {return groupVal_.size(); };

		/** \brief Number of groups
		 *
		 * \return number of groups
		 */
		size_t groupNumber() const {return index_.size(); };

		/** \brief Group ID
		 *
		 * Returns the group ID for a given individual.
		 *
		 * \param[in] ind index of an individual
		 *
		 * \return group ID
		 */
		size_t groupID(const size_t &ind) const {return groupVal_[ind]; };

		/** \brief Update the index
		 *
		 * Updates the groups with a new index. If a group is not present in the new vector, it is left empty but still exists.
		 *
		 * \param[in] newVec new vector of group IDs
		 *
		 */
		void update(const vector<size_t> &newVec);

	private:
		/** \brief Vector of index vectors
		 *
		 * The outside vector is the same length as the number of groups. Each inside vector has the line indexes.
		 */
		vector< vector<size_t> > index_;
		/** \brief Vector of group IDs
		 *
		 * Each element of the vector stores the corresponding (base-0) group ID.
		 */
		vector<size_t> groupVal_;

	};

}

#endif /* index_hpp */
