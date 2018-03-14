#include "caffe/layers/sequence_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void clear_gradients(const int nthreads, int T, int N, int C, Dtype *g, const Dtype *mask) {

	CUDA_KERNEL_LOOP(index, nthreads) {
		int n = (index % (N * C)) / N;
		if (mask[n] == 255) g[index] = 0;
	}
}

template <typename Dtype>
void StandardCTCLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	if (this->phase_ == TRAIN) {
		bool nocont = bottom.size() == 2;
		caffe_gpu_set(bottom[!nocont]->count(), Dtype(0), bottom[!nocont]->mutable_gpu_diff());
		FlattenLabels(bottom[!nocont + 1], nocont ? NULL : bottom[0]);
		Dtype loss = 0;
		if (warpctc) {
			cudaDeviceSynchronize();
			auto options = ctcOptions{};
			options.loc = CTC_GPU;
			CUDA_CHECK(cudaStreamCreate(&(options.stream)));
			options.blank_label = BLANK;

			const Dtype* const activations = bottom[!nocont]->gpu_data();
			Dtype* gradients = bottom[!nocont]->mutable_gpu_diff();
			vector<Dtype> cost(N);
			size_t size_bytes;
			CHECK_CTC_STATUS(get_workspace_size(label_lengths_.data(),
			                                    input_lengths_.data(), C,
			                                    N, options, &size_bytes));
			void* workspace;
			CUDA_CHECK(cudaMalloc(&workspace, size_bytes));
			CHECK_CTC_STATUS(compute_ctc_loss(activations, gradients,
			                                  flat_labels_.data(),
			                                  label_lengths_.data(), input_lengths_.data(),
			                                  C, N, cost.data(),
			                                  workspace, options));
			loss = std::accumulate(cost.begin(), cost.end(), Dtype(0));
			CUDA_CHECK(cudaFree(workspace));
			CUDA_CHECK(cudaStreamDestroy(options.stream));
			CUDA_POST_KERNEL_CHECK;
			if (mask) {
				clear_gradients<Dtype> <<< CAFFE_GET_BLOCKS(T*N*C), CAFFE_CUDA_NUM_THREADS>>>(T * N * C,
				        T, N, C, gradients, mask->gpu_data());
				const Dtype *mask_ptr = mask->cpu_data();
				pos_count = 0;
				for (size_t i = 0; i < N; ++i) {
					if (mask_ptr[i] == 1) {
						++pos_count;
					}
				}
			}
		} else loss = Backward_internal(top, bottom);
		if (!mask) pos_count = N;
		if (pos_count == 0)
			top[0]->mutable_cpu_data()[0] = 0;
		else
			top[0]->mutable_cpu_data()[0] = loss / pos_count;
		caffe_gpu_scal(bottom[!nocont]->count(), Dtype(top[0]->cpu_diff()[0]), bottom[!nocont]->mutable_gpu_diff());
		return;
	}
	Test(bottom, top);
}

template void StandardCTCLayer<float>::Forward_gpu(
    const std::vector<Blob<float>*>& bottom,
    const std::vector<Blob<float>*>& top);
}  // namespace caffe
