# TensorFlow Deformable Conv2D

Custom TensorFlow op for 2D Deformable Convolution (`deform_conv2d`) supporting both Deformable Conv v1 and v2.

- Uses learnable offsets (and masks for v2) to enable dynamic receptive fields.
- CPU implementation via custom C++/TensorFlow op.

## References

- Deformable ConvNets v1: [arXiv:1703.06211](https://arxiv.org/abs/1703.06211)
- Deformable ConvNets v2: [arXiv:1811.11168](https://arxiv.org/abs/1811.11168)

## Files

- `deform_conv2d_kernel.cpp`: Custom CPU kernel implementation
- `deform_conv2d.cpp`: TensorFlow op registration
- `deform_conv2d.py`: Python wrapper for the op
- `setup.py`: Build configuration

## Installation

```bash
python setup.py build_ext --inplace
```

## API

```python
deform_conv2d(
  input,
  weight,
  offset,
  bias=None,
  stride=1,
  padding=0,
  dilation=1,
  mask=None
)
```

### Parameters

- `input`: Tensor `[B, H_in, W_in, C_in]`
- `weight`: Tensor `[K_h, K_w, C_in // groups, C_out]`
- `offset`: Tensor `[B, H_out, W_out, 2 * G * K_h * K_w]`
- `bias`: (optional) Tensor `[C_out]`
- `stride`: `int` or `(int, int)`
- `padding`: `int` or `(int, int)`
- `dilation`: `int` or `(int, int)`
- `mask`: (optional) Tensor `[B, H_out, W_out, G * K_h * K_w]`

### Returns

- Result of convolution: Tensor `[B, H_out, W_out, C_out]`

## Example

```python
>>> import tensorflow as tf
>>> from tf_deform_conv2d import deform_conv2d
>>>
>>> # Sample input
>>> B = 4
>>> H_in, W_in, C_in = 10, 10, 3
>>> H_out, W_out, C_out = 8, 8, 3
>>> kh, kw = 3, 3
>>>
>>> input = tf.random.normal([B, H_in, W_in, C_in])
>>> weight = tf.random.normal([kh, kw, C_in, C_out])
>>>
>>> # offset and mask should have the same spatial size as the output
>>> # of the convolution. In this case, for an input of 10, stride of 1
>>> # and kernel size of 3, without padding, the output size is 8
>>> offset = tf.random.normal([B, H_out, W_out, 2 * kh * kw])
>>> mask = tf.random.normal([B, H_out, W_out, kh * kw])
>>>
>>> output = deform_conv2d(input, weight, offset, mask=mask)
>>> print(tf.shape(output))  # -> [1, H_out, W_out, 64]
>>> # [4, 8, 8, 5]
>>> tf.Tensor([4 8 8 3], shape=(4,), dtype=int32)
```

## Notes

- `mask=None` → Deformable Conv v1
- `mask!=None` → Deformable Conv v2
