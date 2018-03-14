#include "caffe/layers/sequence_layers.hpp"
#include <queue>
#include <map>
#include <unordered_map>

namespace caffe {

template <typename Dtype>
class CTCNode {
public:
	vector <int> _path;
	Dtype _p_path;
	Dtype _p_rest;

	vector <Dtype> _gamma_pb;
	vector <Dtype> _gamma_pn;

	CTCNode(int T):
		_path(0), _p_path(Dtype(0)), _p_rest(Dtype(0)),
		_gamma_pb(T, Dtype(0)), _gamma_pn(T, Dtype(0)) {}

	CTCNode(const CTCNode<Dtype>& ref_):
		_path(ref_._path), _p_path(ref_._p_path), _p_rest(ref_._p_rest),
		_gamma_pb(ref_._gamma_pb), _gamma_pn(ref_._gamma_pn) {}

	Dtype prob_remaining() const {
		return _p_rest;
	}
	Dtype new_label_prob(int k, int t) const {
		return _gamma_pb[t - 1] + ((!_path.empty() && _path.back() == k) ? 0 : _gamma_pn[t - 1]);
	}

	bool operator < (const CTCNode<Dtype>& other_) const {
		return _p_rest < other_._p_rest;
	}
	CTCNode<Dtype>& operator = (const CTCNode<Dtype>& other_) {
		_path = other_._path;
		_p_path = other_._p_path;
		_p_rest = other_._p_rest;
		_gamma_pb = other_._gamma_pb;
		_gamma_pn = other_._gamma_pn;
		return *this;
	}
};

#define ASSIGN(ptr, val) {	\
	*ptr = Dtype(val); 	\
	ptr += N; 		\
}

#define ASSIGN_CHECK(ptr, val) 	\
if (ptr != NULL) {		\
	*ptr = Dtype(val); 	\
	ptr += N; 		\
}

template <typename Dtype>
void StandardCTCLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
	auto param = this->layer_param_.ctc_param();
	thres_above = param.threshold();
	decode_type = param.decode_type();
	warpctc = param.warpctc();
	// thread_num = Caffe::getThreadNum();
	if (decode_type == CTCParameter_Decoder_prefix_lm) {
		thres_cum = param.thres_cum();
		thres_below = param.thres_below();
		alpha = param.alpha();
		beta = param.beta();
		beam_size = param.beam_size();
		lm.Init(param.lm().c_str(), alpha);
	}
	pos_count = -1;
}

template <typename Dtype>
void StandardCTCLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
	T = bottom[0]->num();
	N = bottom[0]->channels();
	vector<int> shape(2);
	shape[0] = T;
	shape[1] = N;
	if ((bottom.size() == 1 && this->phase_ == TEST) || (bottom.size() == 2 && this->phase_ == TRAIN)) {
		// no cont
		C = bottom[0]->height();
	} else {
		// CHECK_EQ(bottom[1]->num(), T);
		CHECK_EQ(bottom[1]->channels(), N);
		C = bottom[1]->height();
		if (bottom.size() == ((this->phase_ == TRAIN) + 3)) {
			mask = bottom.back();
			CHECK_EQ(mask->count(), N);
		} else mask = nullptr;
	}
	gap_per_T = N * C;
	if (this->phase_ == TEST) {
		top[0]->Reshape(shape);
		CHECK_LE(bottom.size(), 3) << "should be 1 or 2 or 3 bottoms in testing";
		if (top.size() >= 2) top[1]->Reshape(shape);
		if (top.size() == 3) top[2]->Reshape(shape);
	} else {
		CHECK_LE(bottom.size(), 4) << "should be 2 or 3 or 4 bottoms in training";
		top[0]->Reshape(vector<int>(1, 1));
		this->set_loss(0, 1);
		top[0]->mutable_cpu_diff()[0] = 1;
	}
}


template <typename Dtype>
void StandardCTCLayer<Dtype>::FlattenLabels(const Blob<Dtype>* label_blob, const Blob<Dtype>* cont_blob) {
	input_lengths_.resize(N, T);
	if (label_blob) {
		label_lengths_.resize(N, 0);
		flat_labels_.clear();
	}
	for (int j = 0; j < N; ++j) {
		if (label_blob) {
			const Dtype * target_seq = label_blob->cpu_data() + j;
			int last_label_count = flat_labels_.size();
			for (int l = 0; l < T && *target_seq != -1;
			        ++l, target_seq += N)
				flat_labels_.push_back(int(*target_seq));
			label_lengths_[j] = flat_labels_.size() - last_label_count;
		}
		if (cont_blob) {
			const Dtype * cont_seq = cont_blob->cpu_data() + j;
			int cont_end = 0;
			int tt = -1;
			while (cont_end < T && *cont_seq != 0) {
				// get next sequence's length
				tt = -*cont_seq;
				int start = cont_end;
				do {
					++cont_end;
					cont_seq += N;
				} while ( cont_end < T && *cont_seq == 1 );
				CHECK_EQ(tt, cont_end - start) << "sequence length should be equal";
				input_lengths_[j] = tt;
			}
		}
	}
}

template <typename Dtype>
void StandardCTCLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	if (this->phase_ == TRAIN) {
		bool nocont = bottom.size() == 2;
		caffe_set(bottom[!nocont]->count(), Dtype(0), bottom[!nocont]->mutable_cpu_diff());
		FlattenLabels(bottom[!nocont + 1], nocont ? NULL : bottom[0]);
		Dtype loss = 0;
		if (warpctc) {
			auto options = ctcOptions{};
			options.loc = CTC_CPU;
			options.num_threads = 1;
			options.blank_label = BLANK;

			const Dtype* const activations = bottom[!nocont]->cpu_data();
			Dtype* gradients = bottom[!nocont]->mutable_cpu_diff();
			vector<Dtype> cost(N);
			size_t size_bytes;
			CHECK_CTC_STATUS(get_workspace_size(label_lengths_.data(),
			                                    input_lengths_.data(), C,
			                                    N, options, &size_bytes));
			char* workspace = new char[size_bytes];

			CHECK_CTC_STATUS(compute_ctc_loss(activations, gradients,
			                                  flat_labels_.data(),
			                                  label_lengths_.data(), input_lengths_.data(),
			                                  C, N, cost.data(),
			                                  workspace, options));
			loss = std::accumulate(cost.begin(), cost.end(), Dtype(0));
			delete[] workspace;
			if (mask) {
				const Dtype *mask_ptr = mask->cpu_data();
				pos_count = 0;
				for (size_t i = 0; i < T * N; ++i) {
					if (mask_ptr[i / T] == 255) {
						caffe_set(C, Dtype(0), gradients + i * C);
					} else if (mask_ptr[i / T] == 1)
						++pos_count;
				}
				pos_count /= T;
			}
		} else loss = Backward_internal(top, bottom);
		if (!mask) pos_count = N;
		if (pos_count == 0)
			top[0]->mutable_cpu_data()[0] = 0;
		else
			top[0]->mutable_cpu_data()[0] = loss / pos_count;
		caffe_scal(bottom[!nocont]->count(), Dtype(top[0]->cpu_diff()[0]), bottom[!nocont]->mutable_cpu_diff());
		return;
	}
	Test(bottom, top);
}

template <typename Dtype>
void StandardCTCLayer<Dtype>::Test(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	caffe_set(T * N, Dtype(-1), top[0]->mutable_cpu_data());
	bool nocont = bottom.size() == 1;
	FlattenLabels(NULL, nocont ? NULL : bottom[0]);
	const Dtype *mask_ptr = mask ? mask->cpu_data() : nullptr;
	for (int j = 0; j < N; ++j) {
		if (mask_ptr && mask_ptr[j] == 255) continue;
		const Dtype * input_seq = bottom[!nocont]->cpu_data() + j * C;
		Dtype * target_seq = top[0]->mutable_cpu_data() + j;
		Dtype *target_indicator = NULL;
		Dtype *target_score = NULL;
		if (top.size() >= 2) target_indicator = top[1]->mutable_cpu_data() + j;
		if (top.size() == 3) target_score = top[2]->mutable_cpu_data() + j;
		switch (decode_type) {
		case CTCParameter_Decoder_best_path:
			BestPathDecode(target_seq, input_seq, input_lengths_[j],
			               target_indicator, target_score);
			break;
		case CTCParameter_Decoder_best_path_thres:
			BestPathThresDecode(target_seq, input_seq, input_lengths_[j],
			                    target_indicator, target_score);
			break;
		case CTCParameter_Decoder_prefix_search:
			PrefixSearchDecode(target_seq, input_seq, input_lengths_[j]);
			break;
		case CTCParameter_Decoder_prefix_lm:
			PrefixLMDecode(target_seq, input_seq, input_lengths_[j]);
			break;
		}
	}
}
template <typename Dtype>
Dtype StandardCTCLayer<Dtype>::Backward_internal(const vector<Blob<Dtype>*>& top,
        const vector<Blob<Dtype>*>& bottom) {
	Dtype loss = 0;
	bool nocont = bottom.size() == 2;
	int offset = 0;
	pos_count = 0;
	const Dtype *mask_ptr = mask ? mask->cpu_data() : nullptr;
	for (int j = 0; j < N; ++j) {
		if (!(mask_ptr && mask_ptr[j] == 255)) {
			const Dtype * input_seq = bottom[!nocont]->cpu_data() + j * C;
			Dtype *bottom_diff = bottom[!nocont]->mutable_cpu_diff() + j * C;
			loss -= AlphaBeta(flat_labels_.data() + offset, label_lengths_[j], input_seq, input_lengths_[j], bottom_diff);
			++pos_count;
		}
		offset += label_lengths_[j];
	}
	return loss;
}

template <typename Dtype>
Dtype StandardCTCLayer<Dtype>::AlphaBeta(
    const int *target, int L, const Dtype * input_seq,
    const int tt, Dtype * bottom_diff) {
	const Dtype scale = 1 / Dtype(N);
	if (L == 0) return 0;
	int ll = 2 * L + 1;
	Dtype loss = 0;
	// calculate the forward variables
	vector<Dtype> alpha(tt * ll, Dtype(0));
	const Dtype * oldFvars;
	Dtype * Fvars = alpha.data();
	Fvars[0] = input_seq[BLANK];
	Fvars[1] = input_seq[target[0]];
	Dtype sum_temp = std::max(Fvars[0] + Fvars[1], EPS);
	Fvars[0] /= sum_temp; Fvars[1] /= sum_temp;
	loss += std::log(sum_temp);
	const Dtype * current_input = input_seq;
	for (int t = 1; t < tt; ++t) {
		const int start = std::max(0, ll - 2 * (tt - t));
		const int end = std::min(2 * t + 2, ll);
		current_input += gap_per_T;
		oldFvars = Fvars;
		Fvars += ll;
		Dtype sum = 0;
		for (int s = start; s < end; ++s) {
			Dtype &fv = Fvars[s];
			if (s & 1) {
				int labelIndex = s / 2;
				int labelNum = target[labelIndex];
				fv = oldFvars[s] + oldFvars[s - 1];
				if (s > 1 && (labelNum != target[labelIndex - 1]) )
					fv += oldFvars[s - 2];
				fv *= current_input[labelNum];
			} else {
				if (s) fv = (oldFvars[s] + oldFvars[s - 1]) * current_input[BLANK];
				else fv = oldFvars[s] * current_input[BLANK];
			}
			sum += fv;
		}
		if (sum < EPS) sum = EPS;
		for (int s = start; s < end; ++s)
			Fvars[s] /= sum;
		loss += std::log(sum);
	}

	// calculate the backward variables
	vector<Dtype> beta(tt * ll, Dtype(0));
	const Dtype * oldBvars;
	Dtype * Bvars = beta.data() + (tt - 1) * ll;
	Bvars[ll - 1] = 0.5;
	Bvars[ll - 2] = 0.5;
	current_input = input_seq + tt * gap_per_T;
	for (int t = tt - 2; t >= 0; --t) {
		const int start = std::max(0, ll - 2 * (tt - t));
		const int end = std::min(2 * t + 2, ll);
		current_input -= gap_per_T;
		oldBvars = Bvars;
		Bvars -= ll;
		Dtype sum = 0;
		for (int s = start; s < end; ++s) {
			Dtype &bv = Bvars[s];
			if (s & 1) {
				int labelIndex = s / 2;
				int labelNum = target[labelIndex];
				bv = oldBvars[s] * current_input[labelNum] +
				     oldBvars[s + 1] * current_input[BLANK];
				if ( (s < (ll - 2)) &&
				        (labelNum != target[labelIndex + 1]) )
					bv += oldBvars[s + 2] * current_input[target[labelIndex + 1]];
			} else {
				bv = oldBvars[s] * current_input[BLANK];
				if ( s < (ll - 1) )
					bv += oldBvars[s + 1] * current_input[target[s / 2]];
			}
			sum += bv;
		}
		if (sum < EPS) sum = EPS;
		for (int s = start; s < end; ++s)
			Bvars[s] /= sum;
	}

	Fvars = alpha.data();
	Bvars = beta.data();
	for (int t = 0; t < tt; ++t, bottom_diff += gap_per_T, input_seq += gap_per_T) {
		vector<Dtype> dEdYTerms(C, Dtype(0));
		Dtype sum = 0;
		for (int s = 0; s < ll; ++s, ++Fvars, ++Bvars) {
			int k = (s & 1) ? target[s / 2] : BLANK;
			Dtype prod = *Fvars * (*Bvars);
			dEdYTerms[k] += prod;
			sum += prod;
		}
		for (int s = 0; s < C; ++s)
			bottom_diff[s] = -scale *
			                 dEdYTerms[s] / std::max(sum * input_seq[s], EPS);
	}
	return loss;
}

template <typename Dtype>
void StandardCTCLayer<Dtype>::BestPathDecode(Dtype * target_seq, const Dtype * y,
        const int tt, Dtype * target_indicator, Dtype * target_score) {
	Dtype best_score = 0;
	int best_ind = -1;
	int last_index = BLANK;
	for (int i = 0; i < tt; ++i) {
		Dtype max_val = -1;
		int max_idx = -1;
		for (int l = 0; l < C; ++l) {
			if (y[l] > max_val) {
				max_val = y[l];
				max_idx = l;
			}
		}
		y += gap_per_T;
		if (max_idx == last_index) {
			if (max_idx != BLANK && max_val > best_score) {
				best_score = max_val;
				best_ind = i;
			}
		} else {
			if (last_index != BLANK) {
				ASSIGN(target_seq, last_index);
				ASSIGN_CHECK(target_indicator, best_ind);
				ASSIGN_CHECK(target_score, best_score);
			}
			last_index = max_idx;
			best_score = max_val;
			best_ind = i;
		}
	}
	if (last_index != BLANK) {
		ASSIGN(target_seq, last_index);
		ASSIGN_CHECK(target_indicator, best_ind);
		ASSIGN_CHECK(target_score, best_score);
	}
}

template <typename Dtype>
void StandardCTCLayer<Dtype>::BestPathThresDecode(Dtype *target_seq, const Dtype *y,
        const int tt, Dtype *target_indicator, Dtype *target_score) {
	// pick max index in a region seperated by blank
	bool save = false;
	Dtype max_val = -1;
	int max_idx = -1;
	for (int i = 0; i < tt; ++i) {
		if (*y >= thres_above) {
			// if this is a blank
			if (save) {
				// if there exists a result, just push it
				ASSIGN(target_seq, max_idx);
				ASSIGN_CHECK(target_indicator, i);
				ASSIGN_CHECK(target_score, max_val);
				max_val = -1;
				max_idx = -1;
				save = false;
			}
		} else {
			for (int l = 1; l < C; ++l) {
				if (y[l] > max_val) {
					max_val = y[l];
					max_idx = l;
				}
			}
			if (i == tt - 1) {
				*target_seq = Dtype(max_idx);
				ASSIGN_CHECK(target_indicator, i);
				ASSIGN_CHECK(target_score, max_val);
			} else save = true;
		}
		y += gap_per_T;
	}
}

template <typename Dtype>
void StandardCTCLayer<Dtype>::PrefixSearchDecode(Dtype *target_seq, const Dtype *y_last,
        const int tt) {
	const Dtype *y = y_last;
	int t_last = 0;
	for (int t = 0; t < tt; ++t, y += gap_per_T) {
		if (*y >= thres_above) {
			if (t != t_last)
				PrefixSearchDecode_inner(target_seq, y_last, t - t_last);
			t_last = t + 1;
			y_last = y + gap_per_T;
		}
	}
	if (t_last != tt)
		PrefixSearchDecode_inner(target_seq, y_last, tt - t_last);
}

template <typename Dtype>
void StandardCTCLayer<Dtype>::PrefixSearchDecode_inner(Dtype *&target_seq, const Dtype *y_org,
        const int tt) {
	map<int, vector<Dtype> > y;   // C * <prob>[T]. Compressed input matrix.
	const Dtype eps = 1e-4;
	vector<Dtype> v(tt, Dtype(0));
	vector<Dtype> yb(tt);
	for (int t = 0; t < tt; t++)
		yb[t] = *(y_org + gap_per_T * t);

	for (int c = 1; c < C; c++) {
		bool required = false;
		for (int t = 0; t < tt; t++) {
			v[t] = *(y_org + gap_per_T * t + c);
			if (v[t] > eps) required = true;
		}
		if (required) y[c] = v;
	}

	std::priority_queue<CTCNode<Dtype> > queue;
	CTCNode<Dtype> p_best(tt);
	p_best._gamma_pb[0] = yb[0];
	for (int t = 1; t < tt; t++)
		p_best._gamma_pb[t] = p_best._gamma_pb[t - 1] * yb[t];

	p_best._p_path = p_best._gamma_pb[tt - 1];
	p_best._p_rest = 1 - p_best._p_path;
	CTCNode<Dtype> l_best(p_best);

	while (p_best._p_rest > l_best._p_path) {
		Dtype prob_remaining = p_best.prob_remaining();
		for (typename map<int, vector<Dtype> >::const_iterator iter = y.begin(); iter != y.end(); ++iter) {
			const int k = iter->first;
			const vector<Dtype>& yk = iter->second;
			CTCNode<Dtype> p(p_best);
			p._path.push_back(k);
			p._gamma_pn[0] = (p_best._path.size() == 0 ? yk[0] : Dtype(0));
			p._gamma_pb[0] = Dtype(0);
			Dtype prefix_prob = p._gamma_pn[0];
			for (int t = 1; t < tt; t++) {
				Dtype new_label_prob = p_best.new_label_prob(k, t);
				p._gamma_pn[t] = yk[t] * (new_label_prob + p._gamma_pn[t - 1]);
				p._gamma_pb[t] = yb[t] * (p._gamma_pb[t - 1] + p._gamma_pn[t - 1]);
				prefix_prob += yk[t] * new_label_prob;
			}
			p._p_path = p._gamma_pn.back() + p._gamma_pb.back();
			p._p_rest = prefix_prob - p._p_path;
			prob_remaining -= p._p_rest;
			if (p._p_path > l_best._p_path) l_best = p;
			if (p._p_rest > l_best._p_path) queue.push(p);
			if (prob_remaining <= l_best._p_path) break;
		}
		if (queue.size() == 0) break;
		p_best = queue.top();
		queue.pop();
	}

	for (int l = 0; l < l_best._path.size(); l++, target_seq += N)
		* target_seq = Dtype(l_best._path[l]);
}

class CTCBeam {
public:
	std::vector<int> _path;
	float _pn[2], _pb[2];
	float _p;

	CTCBeam():
		_path(0), _p(0) {
		_pn[0] = 0;
		_pn[1] = 0;
		_pb[0] = 0;
		_pb[1] = 0;
	}

	CTCBeam(const CTCBeam & ref_):
		_path(ref_._path), _p(ref_._p) {
		_pn[0] = ref_._pn[0];
		_pn[1] = ref_._pn[1];
		_pb[0] = ref_._pb[0];
		_pb[1] = ref_._pb[1];
	}

	float temp_prob(int k, int t_1) const {
		if (!_path.empty() && _path.back() == k)
			return _pb[t_1];
		return _pb[t_1] + _pn[t_1];
	}

	bool operator < (const CTCBeam &_other) const {
		return _p > _other._p;
	}
};

class PathNode {
public:
	PathNode(CTCBeam *ptr = NULL) {data = ptr;}
	~PathNode() {
		for (auto &n : children)
			if (n.second) delete n.second;
		children.clear();
	}
	bool insert(const CTCBeam *ref) {
		const size_t ref_path_size = ref->_path.size();
		PathNode * ptr = this;
		bool flag = false;
		for (size_t i = 0; i < ref_path_size; ++i) {
			int p = ref->_path[i];
			if (ptr->children.find(p) != ptr->children.end()) {
				ptr = ptr->children[p];
			} else {
				do {
					PathNode *tmp = new PathNode();
					p = ref->_path[i];
					ptr->children[p] = tmp;
					ptr = tmp;
					++i;
				} while (i < ref_path_size);
				flag = true;
				break;
			}
		}
		if (flag || ptr->data == NULL) ptr->data = ref;
		return flag;
	}
	PathNode* find_rm_last(const std::vector<int> &ref_path) {
		PathNode * ptr = this;
		for (int i = 0; i < ref_path.size() - 1; ++i) {
			int p = ref_path[i];
			if (ptr->children.find(p) == ptr->children.end()) return NULL;
			ptr = ptr->children[p];
		}
		return ptr;
	}
	void print(std::string prefix) {
		for (auto x : children) {
			printf("%s%zu", prefix.c_str(), x.first);
			x.second->print(prefix + "\t");
		}
		if (data) printf("\n");
	}
	std::unordered_map<size_t, PathNode*> children;
	const CTCBeam* data;
};

template <typename Dtype>
void StandardCTCLayer<Dtype>::PrefixLMDecode(Dtype *&target_seq, const Dtype *y,
        const int tt) {
	vector<Dtype> cum_prob(C, 0);
	vector<int> certain_idx(T, -1);
	// count the cumulative probability of every label in the dictionary
	for (int i = 0; i < C; i++) {
		const Dtype *y_c = y + i;
		Dtype &cum_prob_c = cum_prob[i];
		for (int j = 0; j < tt; j++, y_c += gap_per_T) {
			cum_prob_c += *y_c;
			if (i != 0 && certain_idx[j] == -1 && *y_c > thres_above) {
				certain_idx[j] = i;
			}
		}
	}
	std::set<CTCBeam> B_set[2];
	bool flag = true;
	CTCBeam p_init;
	p_init._pb[0] = 1;
	B_set[0].insert(p_init);
	for (int i = 0; i < tt; ++i, y += gap_per_T) {
		const int t = flag;
		const int t_1 = !flag;
		flag = !flag;
		std::set<CTCBeam> &B_candidate = B_set[t_1];
		std::set<CTCBeam> &B_new = B_set[t];
		B_new.clear();
		Dtype p_blank = y[0];
		PathNode root;
		for (auto &p : B_candidate)
			root.insert(&p);
		for (auto &pt : B_candidate) {
			CTCBeam p(pt);
			Dtype beta_item;
			PathNode *ans = &root;
			int last_label = 0;
			if (!p._path.empty()) {
				last_label = p._path.back();
				p._pn[t] = p._pn[t_1] * y[last_label];
				ans = root.find_rm_last(p._path);
				if (ans && ans->data) p._pn[t] += y[last_label]
					                                  * ans->data->temp_prob(last_label, t_1)
					                                  * lm.languageModelProb(p._path);
				beta_item = pow(Dtype(p._path.size()), beta);
			} else beta_item = 1;
			p._pb[t] = (p._pn[t_1] + p._pb[t_1]) * p_blank;
			p._p = (p._pb[t] + p._pn[t]) * beta_item;
			B_new.insert(p);
			if (B_new.size() > beam_size) B_new.erase(std::prev(B_new.end()));
			if (p_blank > thres_above) continue;
			std::unordered_map<size_t, PathNode*> *node_map = NULL;
			if (p._path.empty()) node_map = &(root.children);
			else if (ans && ans->children.find(last_label) != ans->children.end())
				node_map = &(ans->children[last_label]->children);
			if (certain_idx[i] != -1) {
				size_t k = certain_idx[i];
				if (!(node_map && node_map->find(k) != node_map->end() && (*node_map)[k]->data)) {
					CTCBeam p_k(p);
					p_k._path.push_back(k);
					p_k._pb[t] = 0;
					p_k._pn[t] = y[k]
					             * p.temp_prob(k, t_1)
					             * lm.languageModelProb(p_k._path);
					Dtype beta_item = pow(Dtype(p_k._path.size()), beta);
					p_k._p = p_k._pn[t] * beta_item; // (p_k._pb[t] + p_k._pn[t]) * beta_item
					B_new.insert(p_k);
					if (B_new.size() > beam_size) B_new.erase(std::prev(B_new.end()));
				}
			} else {
				Dtype left_prob = 1 - p_blank;
				for (size_t k = 1; k < C && left_prob >= thres_below; left_prob -= y[k], ++k) {
					if (cum_prob[k] < thres_cum || y[k] < thres_below) continue;
					// repeated
					if (node_map && node_map->find(k) != node_map->end() && (*node_map)[k]->data)
						continue;
					CTCBeam p_k(p);
					p_k._path.push_back(k);
					p_k._pb[t] = 0;
					p_k._pn[t] = y[k]
					             * p.temp_prob(k, t_1)
					             * lm.languageModelProb(p_k._path);
					Dtype beta_item = pow(Dtype(p_k._path.size()), beta);
					p_k._p = p_k._pn[t] * beta_item; // (p_k._pb[t] + p_k._pn[t]) * beta_item
					B_new.insert(p_k);
					if (B_new.size() > beam_size) B_new.erase(std::prev(B_new.end()));
				}
			}
		}
	}
	int t = 0;
	for (auto x : B_set[!flag].begin()->_path) {
		target_seq[t] = x;
		t += N;
	}
	target_seq[t] = -1;
}

char gInstantiationGuardStandardCTCLayer;
template class StandardCTCLayer<float>;
template <typename Dtype>
shared_ptr<Layer<Dtype> > Creator_StandardCTCLayer(const LayerParameter& param)
{
	return shared_ptr<Layer<Dtype> >(new StandardCTCLayer<Dtype>(param));
}
static LayerRegisterer<float> g_creator_f_StandardCTC("StandardCTC", Creator_StandardCTCLayer<float>);

}  // namespace caffe
