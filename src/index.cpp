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

/// Connect lines with populations
/** \file
 * \author Anthony J. Greenberg
 * \copyright Copyright (c) 2018 Anthony J. Greenberg
 * \version 1.0
 *
 * Implementation of a class that relates individuals to groups, similar to an factor in R.
 *
 */

#include <fstream>
#include <vector>
#include <algorithm>

#include "index.hpp"

using std::ifstream;
using std::move;
using std::remove_if;

using namespace BayesicSpace;

/*
 * Index methods
 */

Index::Index(const size_t &Ngroups){
	index_.resize(Ngroups);
}

Index::Index(const size_t *arr, const size_t &N) {
	for (size_t elInd = 0; elInd < N; elInd++) {
		groupVal_.push_back(arr[elInd]);
		if (arr[elInd] >= index_.size()) {
			index_.resize(arr[elInd]+1); // maybe this groupID is a few steps ahead of the current
			index_[arr[elInd]].push_back(elInd);
		}
		else {
			index_[arr[elInd]].push_back(elInd);
		}

	}
	// now remove all empty elements from index_
	index_.erase(
		remove_if(
			index_.begin(), index_.end(), [](const vector<size_t> &v){return v.empty();}
				),
		index_.end()
	);
}

Index::Index(const vector<size_t> &vec) {
	for (size_t elInd = 0; elInd < vec.size(); elInd++) { // need the element indexes so use that instead of an iterator
		groupVal_.push_back(vec[elInd]);
		if (vec[elInd] >= index_.size()) {
			index_.resize(vec[elInd]+1); // maybe this groupID is a few steps ahead of the current
			index_[vec[elInd]].push_back(elInd);
		}
		else {
			index_[vec[elInd]].push_back(elInd);
		}
	}
	// now remove all empty elements from index_
	index_.erase(
		remove_if(
			index_.begin(), index_.end(), [](const vector<size_t> &v){return v.empty();}
				),
		index_.end()
	);
}

Index::Index(const string &inFileName){
	ifstream idxFl(inFileName.c_str());
	string error;

	if (!idxFl) {
		error = "Cannot open file " + inFileName;
		throw error;
	}
	int tmpIn;
	size_t iEl = 0;
	while (idxFl >> tmpIn) {
		if (tmpIn < 0) {
			error = "Negative group ID";
			throw error;
		}
		groupVal_.push_back(tmpIn);
		if (static_cast<size_t>(tmpIn) >= index_.size()) {
			index_.resize(tmpIn+1); // maybe this goupID is a few steps ahead of the current
			index_[tmpIn].push_back(iEl);
			iEl++;
		}
		else {
			index_[tmpIn].push_back(iEl);
			iEl++;
		}
	}
	// now remove all empty elements from index_
	index_.erase(
		remove_if(
			index_.begin(), index_.end(), [](const vector<size_t> &v){return v.empty();}
				),
		index_.end()
	);
}

Index & Index::operator=(const Index &in) {
	if (&in == this) {
		return *this;
	}
	index_    = in.index_;
	groupVal_ = in.groupVal_;
	return *this;
}

Index & Index::operator=(Index &&in){
	if (&in == this) {
		return *this;
	}
	index_    = move(in.index_);
	groupVal_ = move(in.groupVal_);

	return *this;
}

size_t Index::neGroupNumber() const {
	size_t neGN = 0;
	for (auto &g : index_) {
		if ( g.size() ) {
			neGN++;
		}
	}
	return neGN;
}

void Index::update(const vector<size_t> &newVec){
	for (auto &vec : index_) {
		vec.clear();
	}
	for (size_t elInd = 0; elInd < newVec.size(); elInd++) {
		if (newVec[elInd] >= index_.size()) {
			throw string("ERROR: updating vector has an extra group ID in Index::update()");
		} else {
			index_[newVec[elInd]].push_back(elInd);
		}
	}
	groupVal_ = newVec;
}

