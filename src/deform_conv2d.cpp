#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow
{

    using namespace shape_inference;

    Status DeformConv2DShapeFn(InferenceContext *c)
    {
        try
        {

            // std::cout << "[ShapeFn] DeformConv2DShapeFn called" << std::endl;

            // Validate input ranks
            ShapeHandle input_shape;
            TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input_shape)); // [N, C, H, W]
            ShapeHandle kernel_shape;
            TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &kernel_shape));
            ShapeHandle offset_shape;
            TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 4, &offset_shape));
            ShapeHandle mask_shape;
            TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 4, &mask_shape));
            ShapeHandle bias_shape;
            if (c->num_inputs() > 4)
            {
                TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(4), 1, &bias_shape));
            }

            // Extract input dims
            DimensionHandle batch_size = c->Dim(input_shape, 0);
            DimensionHandle in_channels = c->Dim(input_shape, 1);
            DimensionHandle in_height = c->Dim(input_shape, 2);
            DimensionHandle in_width = c->Dim(input_shape, 3);
            DimensionHandle out_channels = c->Dim(kernel_shape, 0);
            DimensionHandle kernel_h = c->Dim(kernel_shape, 2);
            DimensionHandle kernel_w = c->Dim(kernel_shape, 3);

            // std::cout << "  input_shape  : " << c->DebugString(input_shape) << std::endl;
            // std::cout << "  kernel_shape : " << c->DebugString(kernel_shape) << std::endl;
            // std::cout << "  offset_shape : " << c->DebugString(offset_shape) << std::endl;

            // Extract attributes
            int64_t stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, n_offset_grps;
            bool use_mask;
            TF_RETURN_IF_ERROR(c->GetAttr("stride_h", &stride_h));
            TF_RETURN_IF_ERROR(c->GetAttr("stride_w", &stride_w));
            TF_RETURN_IF_ERROR(c->GetAttr("pad_h", &pad_h));
            TF_RETURN_IF_ERROR(c->GetAttr("pad_w", &pad_w));
            TF_RETURN_IF_ERROR(c->GetAttr("dil_h", &dil_h));
            TF_RETURN_IF_ERROR(c->GetAttr("dil_w", &dil_w));
            TF_RETURN_IF_ERROR(c->GetAttr("n_offset_grps", &n_offset_grps));
            TF_RETURN_IF_ERROR(c->GetAttr("use_mask", &use_mask));

            // Output height and width
            DimensionHandle out_height, out_width;
            {
                DimensionHandle kernel_extent_h, temp;
                TF_RETURN_IF_ERROR(c->Subtract(kernel_h, c->MakeDim(1), &temp));
                TF_RETURN_IF_ERROR(c->Multiply(c->MakeDim(dil_h), temp, &kernel_extent_h));
                TF_RETURN_IF_ERROR(c->Add(in_height, c->MakeDim(2 * pad_h), &temp));
                TF_RETURN_IF_ERROR(c->Subtract(temp, kernel_extent_h, &temp));
                TF_RETURN_IF_ERROR(c->Subtract(temp, c->MakeDim(1), &temp));
                TF_RETURN_IF_ERROR(c->Divide(temp, c->MakeDim(stride_h), false, &out_height));
                TF_RETURN_IF_ERROR(c->Add(out_height, c->MakeDim(1), &out_height));
            }

            {
                DimensionHandle kernel_extent_w, temp;
                TF_RETURN_IF_ERROR(c->Subtract(kernel_w, c->MakeDim(1), &temp));
                TF_RETURN_IF_ERROR(c->Multiply(c->MakeDim(dil_w), temp, &kernel_extent_w));
                TF_RETURN_IF_ERROR(c->Add(in_width, c->MakeDim(2 * pad_w), &temp));
                TF_RETURN_IF_ERROR(c->Subtract(temp, kernel_extent_w, &temp));
                TF_RETURN_IF_ERROR(c->Subtract(temp, c->MakeDim(1), &temp));
                TF_RETURN_IF_ERROR(c->Divide(temp, c->MakeDim(stride_w), false, &out_width));
                TF_RETURN_IF_ERROR(c->Add(out_width, c->MakeDim(1), &out_width));
            }

            // Validate offset shape
            DimensionHandle kh_kw, offset_channels;
            TF_RETURN_IF_ERROR(c->Multiply(kernel_h, kernel_w, &kh_kw));
            TF_RETURN_IF_ERROR(c->Multiply(c->MakeDim(2 * n_offset_grps), kh_kw, &offset_channels));
            TF_RETURN_IF_ERROR(c->Merge(c->Dim(offset_shape, 0), batch_size, &batch_size));
            TF_RETURN_IF_ERROR(c->Merge(c->Dim(offset_shape, 1), offset_channels, &offset_channels));
            TF_RETURN_IF_ERROR(c->Merge(c->Dim(offset_shape, 2), out_height, &out_height));
            TF_RETURN_IF_ERROR(c->Merge(c->Dim(offset_shape, 3), out_width, &out_width));

            // Validate mask shape
            if (use_mask && c->num_inputs() > 3)
            {
                DimensionHandle mask_channels;
                TF_RETURN_IF_ERROR(c->Multiply(kh_kw, c->MakeDim(n_offset_grps), &mask_channels));
                TF_RETURN_IF_ERROR(c->Merge(c->Dim(mask_shape, 0), batch_size, &batch_size));
                TF_RETURN_IF_ERROR(c->Merge(c->Dim(mask_shape, 1), mask_channels, &mask_channels));
                TF_RETURN_IF_ERROR(c->Merge(c->Dim(mask_shape, 2), out_height, &out_height));
                TF_RETURN_IF_ERROR(c->Merge(c->Dim(mask_shape, 3), out_width, &out_width));
            }

            // Validate bias
            if (c->num_inputs() > 4 && c->RankKnown(bias_shape) && c->Rank(bias_shape) == 1)
            {
                TF_RETURN_IF_ERROR(c->Merge(c->Dim(bias_shape, 0), out_channels, &out_channels));
            }

            // Final output shape: [N, C_out, H_out, W_out]
            ShapeHandle output_shape = c->MakeShape({batch_size, out_height, out_width, out_channels});
            c->set_output(0, output_shape);

            return ::tensorflow::OkStatus();
        }
        catch (...)
        {
            std::cerr << "[ShapeFn] Exception occurred!" << std::endl;
            return ::tensorflow::errors::Internal("ShapeFn crashed.");
        }
    }

    REGISTER_OP("DeformConv2D")
        .Input("input: T")
        .Input("weight: T")
        .Input("offset: T")
        .Input("mask: T")
        .Input("bias: T")
        .Output("output: T")
        .Attr("T: {float, double}")
        .Attr("stride_h: int")
        .Attr("stride_w: int")
        .Attr("pad_h: int")
        .Attr("pad_w: int")
        .Attr("dil_h: int")
        .Attr("dil_w: int")
        .Attr("n_weight_grps: int")
        .Attr("n_offset_grps: int")
        .Attr("use_mask: bool")
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c)
                    { return DeformConv2DShapeFn(c); });
} // namespace tensorflow