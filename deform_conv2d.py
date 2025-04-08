import math
import os

import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
so_path = os.path.join(BASE_DIR, "src", "deform_conv2d.so")
op = tf.load_op_library(so_path)


def _pair(x):
    return (x, x) if isinstance(x, int) else x


def deform_conv2d(
    input,
    weight,
    offset,
    bias=None,
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    mask=None,
):
    input = tf.transpose(input, [0, 3, 1, 2])
    offset = tf.transpose(offset, [0, 3, 1, 2])
    weight = tf.transpose(weight, [3, 2, 0, 1])

    out_channels = weight.shape[0]
    use_mask = mask is not None

    # Validate and/or create fallback tensors
    if mask is None:
        mask = tf.zeros([tf.shape(input)[0], 1, 1, 1], dtype=input.dtype)
    else:
        mask = tf.transpose(mask, [0, 3, 1, 2])

    if bias is None:
        bias = tf.zeros([out_channels], dtype=input.dtype)

    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    dil_h, dil_w = _pair(dilation)
    weights_h = weight.shape[2]
    weights_w = weight.shape[3]
    n_in_channels = input.shape[1]

    n_offset_grps = offset.shape[1] // (2 * weights_h * weights_w)
    n_weight_grps = n_in_channels // weight.shape[1]

    if n_offset_grps == 0:
        raise RuntimeError(
            "the shape of the offset tensor at dimension 1 is not valid. It should "
            "be a multiple of 2 * weight.size[2] * weight.size[3].\n"
            f"Got offset.shape[1]={offset.shape[1]}, while 2 * weight.size[2] * weight.size[3]={2 * weights_h * weights_w}"
        )

    return op.DeformConv2D(
        input=input,
        weight=weight,
        offset=offset,
        mask=mask,
        bias=bias,
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        dil_h=dil_h,
        dil_w=dil_w,
        n_weight_grps=n_weight_grps,
        n_offset_grps=n_offset_grps,
        use_mask=use_mask,
    )
