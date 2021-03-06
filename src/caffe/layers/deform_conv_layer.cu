#include <vector>

#include "caffe/util/deformable_im2col.hpp"


namespace caffe {

template <typename Dtype>
void DeformConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}
template <typename Dtype>
void DeformConvolutionLayer<Dtype>::deform_forward_gpu_gemm(const Dtype* input,
  const Dtype* weights, Dtype* output, bool skip_im2col,const Dtype* offset ) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_deformable_im2col_gpu(input,offset,col_buffer_.mutable_gpu_data());
    }
    col_buff = col_buffer_.gpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_,
        (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)0., output + output_offset_ * g);
  }
}
template <typename Dtype>
void DeformConvolutionLayer<Dtype>::deform_backward_gpu_gemm(const Dtype*output,const Dtype* data_im,const Dtype* offset,
  const Dtype*weights, Dtype*input_diff,Dtype* offset_diff) {
    Dtype* col_buff = col_buffer_.mutable_gpu_data();
    for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
        conv_out_spatial_dim_, conv_out_channels_ / group_,
        (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
        (Dtype)0., col_buff + col_offset_ * g);
  }

  conv_deformable_col2im_gpu(col_buff,offset,input_diff);
  conv_deformable_col2im_coord_gpu(col_buff,data_im,offset,offset_diff);
}

template <typename Dtype>
void DeformConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  const Dtype* img_data = bottom[0]->gpu_data();
  const Dtype* offset_data = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  for(int n = 0;n< this->num_; ++n){
      deform_forward_gpu_gemm(img_data+n*this->bottom_dim_,weight,top_data+n*this->top_dim_,
        false,offset_data+n*this->offset_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
      }

  }

  /*
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
  */
}

template <typename Dtype>
void DeformConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  const Dtype* img_data = bottom[0]->gpu_data();
  const Dtype* offset_data = bottom[1]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  Dtype* image_diff = bottom[0]->mutable_gpu_diff();
  Dtype* offset_diff = bottom[1]->mutable_gpu_diff();
  const Dtype* top_diff = top[0]->gpu_diff();
    // Bias gradient, if necessary.
  if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
  }
  if (this->param_propagate_down_[0] ) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(img_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        this->deform_backward_gpu_gemm(top_diff+n*this->top_dim_,img_data+n*this->bottom_dim_,offset_data+n*this->offset_dim_,
          weight,image_diff+n*this->bottom_dim_,offset_diff+n*this->offset_dim_);

      }

  /*
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
    */
  }
}

//INSTANTIATE_CLASS(DeformConvolutionLayer);
//REGISTER_LAYER_CLASS(DeformConvolution);
INSTANTIATE_LAYER_GPU_FUNCS(DeformConvolutionLayer);

}  // namespace caffe
