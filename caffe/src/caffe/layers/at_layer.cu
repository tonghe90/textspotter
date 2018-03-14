#include "caffe/util/gpu_util.cuh"
#include "caffe/layers/at_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void set_value_to_constant(const int nthreads, Dtype value, int size,
                                      int i, Dtype* dst) {

	CUDA_KERNEL_LOOP(index, nthreads) {
		dst[index * size + i] = value;
	}
}

template <typename Dtype>
__global__ void copy_values(const int nthreads, int size_src, int k,
                            const Dtype* src, int size_dst, int i, Dtype* dst) {

	CUDA_KERNEL_LOOP(index, nthreads) {
		dst[index * size_dst + i] = src[index * size_src + k];
	}
}

template <typename Dtype>
__global__ void AffineTransformerForwardGPU(const int nthreads, int nPerSrc,
        const Dtype *width, int C,
        int output_H_, int output_W_, int H, int W,
        const Dtype* U, Dtype* V, const Dtype *theta) {

	const int outHW = output_W_ * output_H_;
	CUDA_KERNEL_LOOP(index, nthreads) {
		V[index] = (Dtype)0.;
		const int w = index % output_W_;
		const int h = (index / output_W_) % output_H_;
		const int c = (index / outHW) % C;
		const int n = index / (outHW * C);
		if (!(width && w >= width[n])) {
			const Dtype *theta_n = theta + n * 6;
			const double x = double(w * theta_n[0] + h * theta_n[1] + theta_n[2]);
			const double y = double(w * theta_n[3] + h * theta_n[4] + theta_n[5]);

			Dtype weight;
			const Dtype* pic = U + ((n / nPerSrc) * C + c) * H * W;

			int w_min = max(int(floor(x)), 0);
			int w_max = min(int(ceil(x)), W - 1);
			int h_min = max(int(floor(y)), 0);
			int h_max = min(int(ceil(y)), H - 1);
			for (int hh = h_min; hh <= h_max; ++hh) {
				for (int ww = w_min; ww <= w_max; ++ww) {
					weight = (1 - fabs(x - ww)) * (1 - fabs(y - hh));
					V[index] += weight * pic[hh * W + ww];
				}
			}
		}
	}
}

template <typename Dtype>
void AffineTransformerLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	// set continous
	vector<Dtype *> cont_ptr(top.size() - 1);
	for (size_t i = 1; i < top.size(); ++i) {
		cont_ptr[i - 1] = top[i]->mutable_cpu_data();
		caffe_set(top[i]->count(), Dtype(1), cont_ptr[i - 1]);
		if (const_W_) caffe_set(N, Dtype(-top[i]->shape(0)), cont_ptr[i - 1]);
	}
	if (!const_W_) {
		const Dtype * w = bottom[2]->cpu_data();
		for (size_t i = 0; i < N; ++i) {
			for (int j = 0; j < cont_ptr.size(); ++j) {
				Dtype *current_cont_ptr = cont_ptr[j] + i;
				int length = calcT(float(w[i]), j);
				*current_cont_ptr = -length;
				current_cont_ptr += length * N;
				for (int l = length; l < top[j + 1]->shape(0); ++l) {
					*current_cont_ptr = 0;
					current_cont_ptr += N;
				}
			}
		}
	}

	const Dtype* U = bottom[0]->gpu_data();
	const Dtype* theta = bottom[1]->gpu_data();

	Dtype* V = top[0]->mutable_gpu_data();

	caffe_gpu_set(top[0]->count(), (Dtype)0, V);
	if (pre_defined_count) {
		// compute full_theta
		Dtype *full_theta_data = full_theta.mutable_gpu_data();
		for (int i = 0; i < 6; ++i) {
			if (is_pre_defined_theta[i]) {
				set_value_to_constant<Dtype> <<< CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
				    N, pre_defined_theta[i], 6, i, full_theta_data);
			} else {
				copy_values<Dtype> <<< CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N,
				        6, i, theta, 6, i, full_theta_data);
			}
		}
		theta = full_theta_data;
	}

	const Dtype * w = const_W_ ? NULL : bottom[2]->gpu_data();
	const int nthreads = N * C * output_H_ * output_W_;
	AffineTransformerForwardGPU<Dtype> <<< CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
	    nthreads, nPerSrc, w, C, output_H_, output_W_, H, W, U, V, theta);
}

template <typename Dtype>
__global__ void AffineTransformerBackwardGPU(const int nthreads, int nPerSrc,
        const Dtype *width, int C,
        int output_H_, int output_W_, int H, int W,
        const Dtype* theta, const Dtype* dV, const Dtype* U,
        Dtype* dTheta_tmp_diff, Dtype* dU, bool bp_U, bool bp_theta) {

	const int outHW = output_W_ * output_H_;
	const int offsetN = outHW * C;
	CUDA_KERNEL_LOOP(index, nthreads) {
		const int w = index % output_W_;
		const int h = (index / output_W_) % output_H_;
		const int c = (index / outHW) % C;
		const int n = index / (outHW * C);
		if (!(width && w >= width[n])) {
			const Dtype *theta_n = theta + n * 6;
			const double x = double(w * theta_n[0] + h * theta_n[1] + theta_n[2]);
			const double y = double(w * theta_n[3] + h * theta_n[4] + theta_n[5]);

			Dtype delta_dpx = (Dtype)0.;
			Dtype delta_dpy = (Dtype)0.;

			int pic_off;
			const int offset = ((n / nPerSrc) * C + c) * H * W;
			const Dtype* pic = U + offset;
			Dtype* dpic = dU + offset;
			Dtype weights = 0;
			int w_min = max(int(floor(x)), 0);
			int w_max = min(int(ceil(x)), W - 1);
			int h_min = max(int(floor(y)), 0);
			int h_max = min(int(ceil(y)), H - 1);
			for (int hh = h_min; hh <= h_max; ++hh) {
				for (int ww = w_min; ww <= w_max; ++ww) {
					pic_off = hh * W + ww;
					Dtype tmp_hh = 1 - fabs(y - hh);
					Dtype tmp_ww = 1 - fabs(x - ww);
					if (bp_U) {
						weights = tmp_hh * tmp_ww;
						caffe_gpu_atomic_add(weights * dV[index], dpic + pic_off);
					}
					if (bp_theta) {
						if (ww >= x) delta_dpx += tmp_hh * pic[pic_off];
						else delta_dpx -= tmp_hh * pic[pic_off];
						if (hh >= y) delta_dpy += tmp_ww * pic[pic_off];
						else delta_dpy -= tmp_ww * pic[pic_off];
					}
				}
			}
			delta_dpx *= dV[index];
			delta_dpy *= dV[index];
			if (bp_theta) {
				Dtype *dTheta = dTheta_tmp_diff + offsetN * 6 * n + index % offsetN;
				dTheta[0] += delta_dpx * w;
				dTheta[offsetN * 1] += delta_dpx * h;
				dTheta[offsetN * 2] += delta_dpx;
				dTheta[offsetN * 3] += delta_dpy * w;
				dTheta[offsetN * 4] += delta_dpy * h;
				dTheta[offsetN * 5] += delta_dpy;
			}
		}
	}
}

template <typename Dtype>
void AffineTransformerLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	const Dtype* dV = top[0]->gpu_diff();
	const Dtype* U = bottom[0]->gpu_data();

	Dtype* dTheta_tmp_diff = dTheta_tmp.mutable_gpu_diff();
	Dtype* dU = bottom[0]->mutable_gpu_diff();

	if (propagate_down[0])
		caffe_gpu_set(bottom[0]->count(), (Dtype)0., dU);
	if (propagate_down[1])
		caffe_gpu_set(dTheta_tmp.count(), (Dtype)0., dTheta_tmp_diff);

	const int nthreads = N * C * output_H_ * output_W_;
	const Dtype * theta = pre_defined_count ? full_theta.gpu_data() : bottom[1]->gpu_data();
	const Dtype * w = const_W_ ? NULL : bottom[2]->gpu_data();
	AffineTransformerBackwardGPU<Dtype> <<< CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
	    nthreads, nPerSrc, w, C, output_H_, output_W_, H, W,
	    theta, dV, U, dTheta_tmp_diff, dU, propagate_down[0], propagate_down[1]);

	if (!propagate_down[1]) return;
	Dtype* all_ones_2_data = all_ones_2.mutable_gpu_data();
	caffe_gpu_set(all_ones_2.count(), (Dtype)1., all_ones_2_data);

	Dtype* dTheta = bottom[1]->mutable_gpu_diff();
	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N * 6, 1, output_H_ * output_W_ * C,
	                      (Dtype)1., dTheta_tmp_diff, all_ones_2_data, (Dtype)0., dTheta);

	if (pre_defined_count) {
		for (int i = 0; i < 6; ++i) {
			if (is_pre_defined_theta[i]) {
				set_value_to_constant<Dtype> <<< CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
				    N, 0, 6, i, dTheta);
			}
		}
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(AffineTransformerLayer);

}	// namespace caffe
