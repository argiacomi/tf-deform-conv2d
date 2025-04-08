#define EIGEN_USE_THREADS
#include <cmath>
#include <cstring>
#include <vector>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/platform/byte_order.h"
#include "unsupported/Eigen/CXX11/Tensor"

using namespace tensorflow;

namespace deform_conv2d
{
  namespace
  {

    constexpr int kMaxParallelImgs = 32;

    template <typename T>
    T bilinear_interpolate(
        const T *in,
        int height,
        int width,
        T h,
        T w)
    {
      if (h <= -1 || height <= h || w <= -1 || width <= w)
        return static_cast<T>(0);

      int h_low = static_cast<int>(std::floor(static_cast<float>(h)));
      int w_low = static_cast<int>(std::floor(static_cast<float>(w)));
      int h_high = h_low + 1;
      int w_high = w_low + 1;

      T lh = h - static_cast<T>(h_low);
      T lw = w - static_cast<T>(w_low);
      T hh = static_cast<T>(1.0f) - lh;
      T hw = static_cast<T>(1.0f) - lw;

      T v1 = static_cast<T>(0);
      if (h_low >= 0 && w_low >= 0)
        v1 = in[h_low * width + w_low];
      T v2 = static_cast<T>(0);
      if (h_low >= 0 && w_high <= width - 1)
        v2 = in[h_low * width + w_high];
      T v3 = static_cast<T>(0);
      if (h_high <= height - 1 && w_low >= 0)
        v3 = in[h_high * width + w_low];
      T v4 = static_cast<T>(0);
      if (h_high <= height - 1 && w_high <= width - 1)
        v4 = in[h_high * width + w_high];

      T w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

      T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
      return val;
    }

    template <typename T>
    void deformable_im2col_kernel(
        int n,
        const T *input,
        const T *offset,
        const T *mask,
        int height,
        int width,
        int weight_h,
        int weight_w,
        int pad_h,
        int pad_w,
        int stride_h,
        int stride_w,
        int dilation_h,
        int dilation_w,
        int batch_sz,
        int n_in_channels,
        int n_offset_grps,
        int out_h,
        int out_w,
        bool use_mask,
        T *columns)
    {

      for (int index = 0; index != n; ++index)
      {
        const int out_x = index % out_w;
        const int out_y = (index / out_w) % out_h;
        const int out_b = (index / (out_w * out_h)) % batch_sz;
        const int in_c = index / (out_w * out_h * batch_sz);
        const int out_c = in_c * weight_h * weight_w;

        int c_per_offset_grp = n_in_channels / n_offset_grps;
        const int grp_idx = in_c / c_per_offset_grp;

        auto columns_ptr = columns +
                           (out_c * (batch_sz * out_h * out_w) + out_b * (out_h * out_w) +
                            out_y * out_w + out_x);

        auto input_ptr = input +
                         (out_b * (n_in_channels * height * width) + in_c * (height * width));

        auto offset_ptr = offset +
                          (out_b * n_offset_grps + grp_idx) * 2 * weight_h * weight_w * out_h *
                              out_w;

        const T *mask_ptr = nullptr;
        if (use_mask)
        {
          mask_ptr = mask + (out_b * n_offset_grps + grp_idx) * weight_h * weight_w *
                                out_h * out_w;
        }

        for (int i = 0; i < weight_h; ++i)
        {
          for (int j = 0; j < weight_w; ++j)
          {
            const int mask_idx = i * weight_w + j;
            const int offset_idx = 2 * mask_idx;

            T mask_value = static_cast<T>(1);
            if (use_mask)
            {
              mask_value =
                  mask_ptr[mask_idx * (out_h * out_w) + out_y * out_w + out_x];
            }

            const T offset_h =
                offset_ptr[offset_idx * (out_h * out_w) + out_y * out_w + out_x];
            const T offset_w = offset_ptr
                [(offset_idx + 1) * (out_h * out_w) + out_y * out_w + out_x];
            const T y =
                static_cast<T>((out_y * stride_h - pad_h) + i * dilation_h + offset_h);
            const T x =
                static_cast<T>((out_x * stride_w - pad_w) + j * dilation_w + offset_w);
            *columns_ptr =
                mask_value * bilinear_interpolate(input_ptr, height, width, y, x);
            columns_ptr += batch_sz * out_h * out_w;
          }
        }
      }
    }

    int64_t get_greatest_divisor_below_bound(int64_t n, int64_t bound)
    {
      for (int64_t k = bound; k >= 1; --k)
      {
        if (n % k == 0)
          return k;
      }
      return 1;
    }

    template <typename W, typename C, typename O>
    void addmm_(const W &weight, const C &columns, O &output)
    {
      output.device(Eigen::DefaultDevice()) += weight.contract(columns, Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(1, 0)});
    }

  } // namespace

  template <typename T>
  void deformable_im2col(
      const T *input,
      const T *offset,
      const T *mask,
      int n_in_channels,
      int height,
      int width,
      int weight_h,
      int weight_w,
      int pad_h,
      int pad_w,
      int stride_h,
      int stride_w,
      int dilation_h,
      int dilation_w,
      int out_h,
      int out_w,
      int parallel_imgs,
      int deformable_group,
      bool use_mask,
      T *columns)
  {
    const int num_kernels = n_in_channels * out_h * out_w * parallel_imgs;

    deformable_im2col_kernel<T>(
        num_kernels,
        input,
        offset,
        mask,
        height,
        width,
        weight_h,
        weight_w,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        parallel_imgs,
        n_in_channels,
        deformable_group,
        out_h,
        out_w,
        use_mask,
        columns);
  }

  template <typename T>
  void deform_conv2d_forward_kernel(
      OpKernelContext *context,
      const Tensor &input,
      const Tensor &weight,
      const Tensor &offset,
      const Tensor &mask,
      const Tensor &bias,
      int64_t stride_h,
      int64_t stride_w,
      int64_t pad_h,
      int64_t pad_w,
      int64_t dilation_h,
      int64_t dilation_w,
      int64_t n_weight_grps,
      int64_t n_offset_grps,
      bool use_mask,
      Tensor **out)
  {

    OP_REQUIRES(context, input.dims() == 4,
                errors::InvalidArgument("input must be 4D"));
    OP_REQUIRES(context, offset.dims() == 4,
                errors::InvalidArgument("offset must be 4D"));
    OP_REQUIRES(context, !use_mask || mask.dims() == 4,
                errors::InvalidArgument("mask must be 4D if use_mask is true"));
    OP_REQUIRES(context, weight.dims() == 4,
                errors::InvalidArgument("weight must be 4D"));

    const auto &device = context->eigen_device<Eigen::ThreadPoolDevice>();
    const int64_t batch_sz = input.dim_size(0);
    const int64_t n_in_channels = input.dim_size(1);
    const int64_t in_h = input.dim_size(2);
    const int64_t in_w = input.dim_size(3);

    const int64_t n_parallel_imgs = get_greatest_divisor_below_bound(batch_sz, kMaxParallelImgs);

    const int64_t out_channels = weight.dim_size(0);
    const int64_t weight_h = weight.dim_size(2);
    const int64_t weight_w = weight.dim_size(3);

    const int64_t ker_h = dilation_h * (weight_h - 1) + 1;
    const int64_t ker_w = dilation_w * (weight_w - 1) + 1;
    const int64_t out_h = (in_h + 2 * pad_h - ker_h) / stride_h + 1;
    const int64_t out_w = (in_w + 2 * pad_w - ker_w) / stride_w + 1;

    OP_REQUIRES(context, weight_h > 0 && weight_w > 0,
                errors::InvalidArgument("Invalid weight dims: weight_h=", weight_h, ", weight_w=", weight_w));
    OP_REQUIRES(context, stride_h > 0 && stride_w > 0,
                errors::InvalidArgument("Invalid stride: stride_h=", stride_h, ", stride_w=", stride_w));
    OP_REQUIRES(context, pad_h >= 0 && pad_w >= 0,
                errors::InvalidArgument("Invalid padding: pad_h=", pad_h, ", pad_w=", pad_w));
    OP_REQUIRES(context, dilation_h > 0 && dilation_w > 0,
                errors::InvalidArgument("Invalid dilation: dilation_h=", dilation_h, ", dilation_w=", dilation_w));
    OP_REQUIRES(context, weight.dim_size(1) * n_weight_grps == input.dim_size(1),
                errors::InvalidArgument("Mismatch: weight.size(1) * n_weight_grps != input.size(1)"));
    OP_REQUIRES(context, weight.dim_size(0) % n_weight_grps == 0,
                errors::InvalidArgument("weight.size(0) must be divisible by n_weight_grps"));
    OP_REQUIRES(context, offset.dim_size(1) == n_offset_grps * 2 * weight_h * weight_w,
                errors::InvalidArgument("Invalid offset.shape[1]: got ", offset.dim_size(1),
                                        ", expected ", n_offset_grps * 2 * weight_h * weight_w));
    OP_REQUIRES(context, !use_mask || mask.dim_size(1) == n_offset_grps * weight_h * weight_w,
                errors::InvalidArgument("Invalid mask.shape[1]: got ", mask.dim_size(1),
                                        ", expected ", n_offset_grps * weight_h * weight_w));
    OP_REQUIRES(context, input.dim_size(1) % n_offset_grps == 0,
                errors::InvalidArgument("input.size(1) must be divisible by n_offset_grps"));
    OP_REQUIRES(context, offset.dim_size(0) == input.dim_size(0),
                errors::InvalidArgument("Batch size mismatch: offset vs input"));
    OP_REQUIRES(context,
                offset.dim_size(2) == out_h && offset.dim_size(3) == out_w,
                errors::InvalidArgument("offset dims mismatch: got (", offset.dim_size(2), ", ", offset.dim_size(3),
                                        "), expected (", out_h, ", ", out_w, ")"));

    if (use_mask)
    {
      OP_REQUIRES(context, mask.dim_size(0) == input.dim_size(0),
                  errors::InvalidArgument("Batch size mismatch: mask vs input"));
      OP_REQUIRES(context,
                  mask.dim_size(2) == out_h && mask.dim_size(3) == out_w,
                  errors::InvalidArgument("mask dims mismatch: got (", mask.dim_size(2), ", ", mask.dim_size(3),
                                          "), expected (", out_h, ", ", out_w, ")"));
    }
    OP_REQUIRES(context, out_h > 0 && out_w > 0,
                errors::InvalidArgument("Output size too small: out_h=", out_h, ", out_w=", out_w));

    TensorShape out_shape({batch_sz, out_h, out_w, out_channels});
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, out));
    if (batch_sz == 0)
      return;

    const auto &input_reshaped = input.shaped<T, 5>({batch_sz / n_parallel_imgs, n_parallel_imgs, n_in_channels, in_h, in_w});
    const auto &offset_reshaped = offset.shaped<T, 5>({batch_sz / n_parallel_imgs, n_parallel_imgs, n_offset_grps * 2 * weight_h * weight_w, out_h, out_w});

    TensorShape buf_shape({batch_sz / n_parallel_imgs, out_channels, n_parallel_imgs * out_h, out_w});
    Tensor out_buf;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::v(), buf_shape, &out_buf));

    out_buf.flat<T>().device(device) = out_buf.flat<T>().constant(static_cast<T>(0));

    auto out_buf_reshaped = out_buf.shaped<T, 5>({batch_sz / n_parallel_imgs, n_weight_grps, out_channels / n_weight_grps, n_parallel_imgs * out_h, out_w});
    auto weight_reshaped = weight.shaped<T, 5>({n_weight_grps, out_channels / n_weight_grps, n_in_channels / n_weight_grps, weight_h, weight_w});

    Tensor columns;
    TensorShape col_shape({n_in_channels * weight_h * weight_w, n_parallel_imgs * out_h * out_w});
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::v(), col_shape, &columns));

    const auto *input_data = input.flat<T>().data();
    const auto *offset_data = offset.flat<T>().data();
    const T *mask_data = use_mask ? mask.flat<T>().data() : nullptr;

    for (int b = 0; b < batch_sz / n_parallel_imgs; ++b)
    {
      int64_t input_offset = b * (n_parallel_imgs * n_in_channels * in_h * in_w);
      int64_t offset_offset = b * (n_parallel_imgs * n_offset_grps * 2 * weight_h * weight_w * out_h * out_w);
      int64_t mask_offset = use_mask ? (b * (n_parallel_imgs * n_offset_grps * weight_h * weight_w * out_h * out_w)) : 0;

      const T *cur_input_ptr = input_data + input_offset;
      const T *cur_offset_ptr = offset_data + offset_offset;
      const T *cur_mask_ptr = use_mask ? (mask_data + mask_offset) : nullptr;

      deformable_im2col(
          cur_input_ptr,
          cur_offset_ptr,
          cur_mask_ptr,
          n_in_channels,
          in_h,
          in_w,
          weight_h,
          weight_w,
          pad_h,
          pad_w,
          stride_h,
          stride_w,
          dilation_h,
          dilation_w,
          out_h,
          out_w,
          n_parallel_imgs,
          n_offset_grps,
          use_mask,
          columns.flat<T>().data());

      auto columns_grouped = columns.shaped<T, 3>({n_weight_grps, columns.dim_size(0) / n_weight_grps, columns.dim_size(1)});

      int dim0 = out_channels / n_weight_grps;
      int dim1 = (n_in_channels / n_weight_grps) * weight_h * weight_w;
      int dim2 = n_parallel_imgs * out_h * out_w;

      for (int g = 0; g < n_weight_grps; ++g)
      {
        auto weight_slice_op = weight_reshaped.template chip<0>(g);
        auto weight_mat = weight_slice_op.reshape(
            Eigen::array<Eigen::Index, 2>{dim0, dim1});

        auto col_slice_op = columns_grouped.template chip<0>(g);
        auto col_mat = col_slice_op.reshape(
            Eigen::array<Eigen::Index, 2>{dim1, dim2});

        auto out_slice_op = out_buf_reshaped.template chip<0>(g);
        auto out_mat = out_slice_op.reshape(
            Eigen::array<Eigen::Index, 2>{dim0, dim2});

        addmm_(weight_mat, col_mat, out_mat);
      }
    }

    auto out_buf_final = out_buf.shaped<T, 6>({batch_sz / n_parallel_imgs, n_weight_grps, out_channels / n_weight_grps, n_parallel_imgs, out_h, out_w});
    auto out_buf_shuffled_imgs = out_buf_final.shuffle(Eigen::array<int, 6>{0, 3, 1, 2, 4, 5});
    auto nchw_output = out_buf_shuffled_imgs.reshape(Eigen::DSizes<int, 4>{static_cast<int>(batch_sz), static_cast<int>(out_channels), static_cast<int>(out_h), static_cast<int>(out_w)});

    auto bias_reshaped = bias.shaped<T, 4>({1, out_channels, 1, 1});
    Eigen::DSizes<int, 4> bcast{static_cast<int>(batch_sz), 1, static_cast<int>(out_h), static_cast<int>(out_w)};
    nchw_output.device(device) += bias_reshaped.broadcast(bcast);

    auto nhwc_output = (*out)->shaped<T, 4>({batch_sz, out_h, out_w, out_channels});
    nhwc_output.device(device) = nchw_output.shuffle(Eigen::array<int, 4>{0, 2, 3, 1});

    return;
  }

  template <typename T>
  class DeformConv2DOp : public OpKernel
  {
  public:
    explicit DeformConv2DOp(OpKernelConstruction *context) : OpKernel(context)
    {
      OP_REQUIRES_OK(context, context->GetAttr("stride_h", &stride_h));
      OP_REQUIRES_OK(context, context->GetAttr("stride_w", &stride_w));
      OP_REQUIRES_OK(context, context->GetAttr("pad_h", &pad_h));
      OP_REQUIRES_OK(context, context->GetAttr("pad_w", &pad_w));
      OP_REQUIRES_OK(context, context->GetAttr("dil_h", &dil_h));
      OP_REQUIRES_OK(context, context->GetAttr("dil_w", &dil_w));
      OP_REQUIRES_OK(context, context->GetAttr("n_weight_grps", &n_weight_grps));
      OP_REQUIRES_OK(context, context->GetAttr("n_offset_grps", &n_offset_grps));
      OP_REQUIRES_OK(context, context->GetAttr("use_mask", &use_mask_));
    }

    void Compute(OpKernelContext *context) override
    {
      const Tensor &input = context->input(0);
      const Tensor &weight = context->input(1);
      const Tensor &offset = context->input(2);
      const Tensor &mask = context->input(3);
      const Tensor &bias = context->input(4);

      Tensor *output = nullptr;

      deform_conv2d_forward_kernel<T>(
          context,
          input,
          weight,
          offset,
          mask,
          bias,
          stride_h,
          stride_w,
          pad_h,
          pad_w,
          dil_h,
          dil_w,
          n_weight_grps,
          n_offset_grps,
          use_mask_,
          &output);
    }

  private:
    int stride_h;
    int stride_w;
    int pad_h;
    int pad_w;
    int dil_h;
    int dil_w;
    int n_weight_grps;
    int n_offset_grps;
    bool use_mask_;
  };

} // namespace deform_conv2d

#define REGISTER_DEFORM_CONV_KERNEL(T)                                \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("DeformConv2D").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      ::deform_conv2d::DeformConv2DOp<T>);

TF_CALL_float(REGISTER_DEFORM_CONV_KERNEL);
TF_CALL_double(REGISTER_DEFORM_CONV_KERNEL);
TF_CALL_half(REGISTER_DEFORM_CONV_KERNEL);
TF_CALL_bfloat16(REGISTER_DEFORM_CONV_KERNEL);
