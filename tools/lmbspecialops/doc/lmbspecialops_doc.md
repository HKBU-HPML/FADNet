# lmbspecialops documentation

Op name | Summary
--------|--------
[decode_flo](#decode_flo) | Decode a FLO-encoded image to a float tensor.
[decode_lz4](#decode_lz4) | Decode LZ4-encoded data.
[decode_lz4_raw](#decode_lz4_raw) | Decode LZ4-encoded data.
[decode_pfm](#decode_pfm) | Decode a PFM-encoded image to a uint8 or uint16 tensor.
[decode_ppm](#decode_ppm) | Decode a PPM-encoded image to a uint8 or uint16 tensor.
[decode_webp](#decode_webp) | Decode a WEBP-encoded image to a uint8 or uint16 tensor.
[depth_to_flow](#depth_to_flow) | Computes the optical flow for an image pair based on the depth map and camera motion.
[depth_to_normals](#depth_to_normals) | Computes the normal map from a depth map.
[encode_flo](#encode_flo) | Encode float flow data as FLO file.
[encode_lz4](#encode_lz4) | Encode LZ4-encoded data.
[encode_lz4_raw](#encode_lz4_raw) | Encode LZ4-encoded data.
[encode_pfm](#encode_pfm) | Encode float flow data as PFM file.
[encode_webp](#encode_webp) | Encode uint8 image data as WEBP file.
[flow_to_depth](#flow_to_depth) | DEPRECATED Computes the depth from optical flow and the camera motion. DEPRECATED
[flow_to_depth2](#flow_to_depth2) | Computes the depth from optical flow and the camera motion.
[leaky_relu](#leaky_relu) | Computes the leaky rectified linear unit activations y = max(leak*x,x).
[median3x3_downsample](#median3x3_downsample) | Downsamples an image with a 3x3 median filter with a stride of 2.
[normalized_differences](#normalized_differences) | This op computes normalized differences between the center pixel and a list of neighbour pixels.
[replace_nonfinite](#replace_nonfinite) | Replaces all nonfinite elements.
[scale_invariant_gradient](#scale_invariant_gradient) | This op computes the scale invariant spatial gradient as described in the DeMoN paper.
[warp2d](#warp2d) | Warps the input with the given displacement vector field.

## decode_flo

```python
decode_flo(contents)
```

Decode a FLO-encoded image to a float tensor.



*There is no corresponding gradient op*

#### Args

* ```contents```: 0-D.  The FLO-encoded image.



#### Returns

3-D with shape `[height, width, 2]`.

## decode_lz4

```python
decode_lz4(contents, expected_shape=)
```

Decode LZ4-encoded data.



*There is no corresponding gradient op*

#### Args

* ```contents```: 0-D. The LZ4-encoded data.

* ```expected_shape```: Expected shape of the output.



#### Returns

1D decoded data of type dtype.

## decode_lz4_raw

```python
decode_lz4_raw(contents, expected_size=0)
```

Decode LZ4-encoded data.



*There is no corresponding gradient op*

#### Args

* ```contents```: 0-D. The LZ4-encoded data.

* ```expected_size```: Expected size of the output.



#### Returns

0-D. Decoded data.

## decode_pfm

```python
decode_pfm(contents, channels=0)
```

Decode a PFM-encoded image to a uint8 or uint16 tensor.



*There is no corresponding gradient op*

#### Args

* ```contents```: 0-D.  The PFM-encoded image.

* ```channels```: Number of color channels for the decoded image.



#### Returns

3-D with shape `[height, width, {3|1}]`.Scalar containing the scale.

## decode_ppm

```python
decode_ppm(contents)
```

Decode a PPM-encoded image to a uint8 or uint16 tensor.



*There is no corresponding gradient op*

#### Args

* ```contents```: 0-D.  The PPM-encoded image.



#### Returns

3-D with shape `[height, width, 3]`.maxval from the ppm file.

## decode_webp

```python
decode_webp(contents, channels=0)
```

Decode a WEBP-encoded image to a uint8 or uint16 tensor.



*There is no corresponding gradient op*

#### Args

* ```contents```: 0-D.  The WEBP-encoded image.

* ```channels```: Number of expected channels (0:auto|3|4).



#### Returns

uint8 tensor with shape `[height, width, {3|4}]`.

## depth_to_flow

```python
depth_to_flow(depth, intrinsics, rotation, translation, rotation_format='angleaxis3', inverse_depth=False, normalize_flow=False)
```

Computes the optical flow for an image pair based on the depth map and camera motion.

Takes the depth map of the first image and the relative camera motion of the
second image and computes the optical flow from the first to the second image.
The op assumes that the internal camera parameters are the same for both cameras.

*There is no corresponding gradient op*

#### Args

* ```depth```: depth map with absolute or inverse depth values
The depth values describe the z distance to the optical center.

* ```intrinsics```: camera intrinsics in the format [fx, fy, cx, cy].
fx,fy are the normalized focal lengths.
cx,cy is the normalized position of the principal point.

* ```rotation```: The relative rotation R of the second camera.
RX+t transforms a point X to the camera coordinate system of the second camera

* ```translation```: The relative translation vector t of the second camera.
RX+t transforms a point X to the camera coordinate system of the second camera

* ```rotation_format```: The format for the rotation.
Allowed values are {'matrix', 'quaternion', 'angleaxis3'}.
'matrix' is a 3x3 rotation matrix in column major order
'quaternion' is a quaternion given as [w,x,y,z], w is the coefficient for the real part.
'angleaxis3' is a 3d vector with the rotation axis. The angle is encoded as magnitude.

* ```inverse_depth```: If true then the input depth map must use inverse depth values.

* ```normalize_flow```: If true the returned optical flow will be normalized with respect to the
image dimensions.



#### Returns


A tensor with the optical flow from the first to the second image.
The format of the output tensor is NCHW with C=2; [batch, 2, height, width].

## depth_to_normals

```python
depth_to_normals(depth, intrinsics, inverse_depth=False)
```

Computes the normal map from a depth map.



*There is no corresponding gradient op*

#### Args

* ```depth```: depth map with absolute or inverse depth values
The depth values describe the z distance to the optical center.

* ```intrinsics```: camera intrinsics in the format [fx, fy, cx, cy].
fx,fy are the normalized focal lengths.
cx,cy is the normalized position of the principal point.

* ```inverse_depth```: If true then the input depth map must use inverse depth values.



#### Returns


Normal map in the coordinate system of the camera.
The format of the output tensor is NCHW with C=3; [batch, 3, height, width].

## encode_flo

```python
encode_flo(image)
```

Encode float flow data as FLO file.



*There is no corresponding gradient op*

#### Args

* ```image```: 3-D with shape `[height, width, 2]`.



#### Returns

0-D.  The FLO-encoded data.

## encode_lz4

```python
encode_lz4(contents)
```

Encode LZ4-encoded data.



*There is no corresponding gradient op*

#### Args

* ```contents```: n-D. The data.



#### Returns

0-D. Encoded data.

## encode_lz4_raw

```python
encode_lz4_raw(contents)
```

Encode LZ4-encoded data.



*There is no corresponding gradient op*

#### Args

* ```contents```: 0-D. The data.



#### Returns

0-D. Encoded data.

## encode_pfm

```python
encode_pfm(image, scale=-1.0)
```

Encode float flow data as PFM file.



*There is no corresponding gradient op*

#### Args

* ```image```: 3-D with shape `[height, width, {1|3}]`.

* ```scale```: Scalar containing the scale. Negative for little endian encoding.



#### Returns

0-D.  The PFM-encoded data.

## encode_webp

```python
encode_webp(image, lossless=False, lossless_level=6, preset='default', preset_quality=7.5e+01, quality=-1.0, method=-1, image_hint='unspecified', target_size=-1, target_PSNR=0.0, segments=-1, sns_strength=-1, filter_strength=-1, filter_sharpness=-1, filter_type=-1, autofilter=-1, alpha_compression=-1, alpha_filtering=-1, alpha_quality=-1, pass_=-1, show_compressed=False, preprocessing=-1, partitions=0, partition_limit=-1, emulate_jpeg_size=False, thread_level=False, low_memory=False, near_lossless=100, exact=False)
```

Encode uint8 image data as WEBP file.



*There is no corresponding gradient op*

#### Args

* ```image```: uint8 tensor with shape `[height, width, {3|4}]`.

* ```lossless```: bool, Lossless (or lossy) encoding. Default is false.

* ```lossless_level```: int, Activate the lossless compression mode with the desired efficiency level between 0 (fastest,
lowest compression) and 9 (slower, best compression). A good default level is '6', providing a fair tradeoff between
compression speed and final compressed size.

* ```preset```: int, used to initialize the encoder settings:
'default': Default preset.
'picture': Picture: Digital picture, like portrait, inner shot
'photo': Photo: Outdoor photograph, with natural lighting
'drawing': Drawing: Hand or line drawing, with high-contrast details
'icon': Icon. Small-sized colorful images
'text': Text: Text-like

* ```preset_quality```: float, between 0 (smallest file) and 100 (biggest), used to initialize the encoder settings.

Encoder configuration:

* ```quality```: float, between 0 (smallest file) and 100 (biggest)

* ```method```: int, quality/speed trade-off (0=fast, 6=slower-better)

* ```image_hint```: int, Hint for image type (lossless only for now):
  'default': Default preset.
  'picture': Digital picture, like portrait, inner shot
  'photo': Outdoor photograph, with natural lighting
  'graph': Discrete tone image (graph, map-tile etc).
  'unspecified': Keep from preset.

Parameters related to lossy compression only:

* ```target_size```: int, if non-zero, set the desired target size in bytes. Takes precedence over the 'compression' parameter.

* ```target_PSNR```: float, if non-zero, specifies the minimal distortion to try to achieve. Takes precedence over target_size.

* ```segments```: int, maximum number of segments to use, in [1..4]

* ```sns_strength```: int, Spatial Noise Shaping. 0=off, 100=maximum.

* ```filter_strength```: int, range: [0 = off .. 100 = strongest]

* ```filter_sharpness```: int, range: [0 = off .. 7 = least sharp]

* ```filter_type```: int, filtering type: 0 = simple, 1 = strong (only used if filter_strength > 0 or autofilter > 0)

* ```autofilter```: int, Auto adjust filter's strength [0 = off, 1 = on]

* ```alpha_compression```: int, Algorithm for encoding the alpha plane (0 = none, 1 = compressed with WebP lossless). Default is 1.

* ```alpha_filtering```: int, Predictive filtering method for alpha plane. 0: none, 1: fast, 2: best. Default if 1.

* ```alpha_quality```: int, Between 0 (smallest size) and 100 (lossless). Default is 100.

* ```pass_```: int, number of entropy-analysis passes (in [1..10]).

* ```show_compressed```: bool, if set, export the compressed picture back. In-loop filtering is not applied.

* ```preprocessing```: int, preprocessing filter:
0=none, 1=segment-smooth, 2=pseudo-random dithering

* ```partitions```: int, log2(number of token partitions) in [0..3]. Default is set to 0 for easier progressive decoding.

* ```partition_limit```: int, quality degradation allowed to fit the 512k limit on prediction modes coding (0: no degradation,
100: maximum possible degradation).

* ```emulate_jpeg_size```: bool, If true, compression parameters will be remapped to better match the expected output size from
JPEG compression. Generally, the output size will be similar but the degradation will be lower.

* ```thread_level```: bool, If set, try and use multi-threaded encoding.

* ```low_memory```: bool, If set, reduce memory usage (but increase CPU use).

* ```near_lossless```: int, Near lossless encoding [0 = max loss .. 100 = off (default)].

* ```exact```: bool, if set, preserve the exact RGB values under transparent area. Otherwise, discard this invisible RGB
information for better compression. The default value is false.



#### Returns

0-D.  The WEBP-encoded data.

## flow_to_depth

```python
flow_to_depth(flow, intrinsics, rotation, translation, rotation_format='angleaxis3', inverse_depth=False, normalized_flow=False)
```

DEPRECATED Computes the depth from optical flow and the camera motion. DEPRECATED

The behaviour of this function is incorrect. This function is kept for
compatibility with old networks, which rely on this specific behaviour.
For new code use flow_to_depth2() instead.

*There is no corresponding gradient op*

#### Args

* ```flow```: optical flow normalized or in pixel units. The tensor format must be NCHW.

* ```intrinsics```: camera intrinsics in the format [fx, fy, cx, cy].
fx,fy are the normalized focal lengths.
cx,cy is the normalized position of the principal point.
The format of the tensor is NC with C=4 and N matching the batch size of the
flow tensor.

* ```rotation```: The relative rotation R of the second camera.
RX+t transforms a point X to the camera coordinate system of the second camera
The format of the tensor is either NC with C=3 or C=4 or NIJ with I=3 and J=3.
N matches the batch size of the flow tensor.

* ```translation```: The relative translation vector t of the second camera.
RX+t transforms a point X to the camera coordinate system of the second camera
The format of the tensor is NC with C=3 and N matching the batch size of the
flow tensor.

* ```rotation_format```: The format for the rotation.
Allowed values are {'matrix', 'quaternion', 'angleaxis3'}.
'angleaxis3' is a 3d vector with the rotation axis.
The angle is encoded as magnitude.

* ```inverse_depth```: If true then the output depth map uses inverse depth values.

* ```normalized_flow```: If true then the input flow is expected to be normalized with respect to the
image dimensions.



#### Returns


A tensor with the depth for the first image.
The format of the output tensor is NCHW with C=1; [batch, 1, height, width].

## flow_to_depth2

```python
flow_to_depth2(flow, intrinsics, rotation, translation, rotation_format='angleaxis3', inverse_depth=False, normalized_flow=False)
```

Computes the depth from optical flow and the camera motion.

Takes the optical flow and the relative camera motion from the second camera to
compute a depth map.
The layer assumes that the internal camera parameters are the same for both
images.

*There is no corresponding gradient op*

#### Args

* ```flow```: optical flow normalized or in pixel units. The tensor format must be NCHW.

* ```intrinsics```: camera intrinsics in the format [fx, fy, cx, cy].
fx,fy are the normalized focal lengths.
cx,cy is the normalized position of the principal point.
The format of the tensor is NC with C=4 and N matching the batch size of the
flow tensor.

* ```rotation```: The relative rotation R of the second camera.
RX+t transforms a point X to the camera coordinate system of the second camera
The format of the tensor is either NC with C=3 or C=4 or NIJ with I=3 and J=3.
N matches the batch size of the flow tensor.

* ```translation```: The relative translation vector t of the second camera.
RX+t transforms a point X to the camera coordinate system of the second camera
The format of the tensor is NC with C=3 and N matching the batch size of the
flow tensor.

* ```rotation_format```: The format for the rotation.
Allowed values are {'matrix', 'quaternion', 'angleaxis3'}.
'angleaxis3' is a 3d vector with the rotation axis.
The angle is encoded as magnitude.

* ```inverse_depth```: If true then the output depth map uses inverse depth values.

* ```normalized_flow```: If true then the input flow is expected to be normalized with respect to the
image dimensions.



#### Returns


A tensor with the depth for the first image.
The format of the output tensor is NCHW with C=1; [batch, 1, height, width].

## leaky_relu

```python
leaky_relu(input, leak=0.1)
```

Computes the leaky rectified linear unit activations y = max(leak*x,x).



*This op has a corresponding gradient implementation*

#### Args

* ```input```: Input tensor of any shape.

* ```leak```: The leak factor.



#### Returns


A tensor with the activation.

## median3x3_downsample

```python
median3x3_downsample(input)
```

Downsamples an image with a 3x3 median filter with a stride of 2.



*There is no corresponding gradient op*

#### Args

* ```input```: Tensor with at least rank 2.
The supported format is NCHW [batch, channels, height, width].



#### Returns


Downsampled tensor.

## normalized_differences

```python
normalized_differences(input, offsets=[1,0,0,1], weights=[1.0,1.0], epsilon=0.001)
```

This op computes normalized differences between the center pixel and a list of neighbour pixels.

The normalized difference is computed as
  d(x) = w*(u(x+v) - u(x))/(|u(x+v)| + |u(x)| + eps),
with
 x   as 2d position (x,y)
 v   as 2d offset vector (vx,vy)
 u   as input channel
 w   as weight
 eps as epsilon value


Note that this op does not distinguish between channels and batch size of the
input tensor. If the input tensor has more than one channel, then the resulting
batch size will be the product of the input batch size and the channels.
The number of output channels (co) is the number of offset vectors given.
E.g. (bi,ci,hi,wi) -> (bi*ci, co, hi, wi).

*This op has a corresponding gradient implementation*

#### Args

* ```input```: An input tensor with at least rank 2.

* ```offsets```: The offset vector for the neighbouring pixel.
Two consecutive elements form a 2d vector.

* ```weights```: The weight factor for each difference.
The size of this vector is half the size of 'offsets'.

* ```epsilon```: epsilon value for avoiding division by zero



#### Returns


Tensor with the normalized differences.
The format of the output tensor is NCHW with C=len(weights); [batch, channels, height, width].

## replace_nonfinite

```python
replace_nonfinite(input, value=0.0)
```

Replaces all nonfinite elements.

Replaces nonfinite elements in a tensor with a specified value.
The corresponding gradient for replaced elements is 0.

*This op has a corresponding gradient implementation*

#### Args

* ```input```: Input tensor of any shape.

* ```value```: The value used for replacing nonfinite elements.



#### Returns


Tensor with all nonfinite values replaced with 'value'.

## scale_invariant_gradient

```python
scale_invariant_gradient(input, deltas=[1], weights=[1.0], epsilon=0.001)
```

This op computes the scale invariant spatial gradient as described in the DeMoN paper.

The x component is computed as:
  grad_x = sum_delta w*(u(x+delta,y) - u(x,y))/(|u(x+delta,y)| + |u(x,y)| + eps)

Note that this op does not distinguish between channels and batch size of the
input tensor. If the input tensor has more than one channel, then the resulting
batch size will be the product of the input batch size and the channels.
E.g. (bi,ci,hi,wi) -> (bi*ci, 2, h, w).

*This op has a corresponding gradient implementation*

#### Args

* ```input```: An input tensor with at least rank 2.

* ```deltas```: The pixel delta for the difference.
This vector must be the same length as weight.

* ```weights```: The weight factor for each difference.
This vector must be the same length as delta.

* ```epsilon```: epsilon value for avoiding division by zero



#### Returns


Tensor with the scale invariant spatial gradient.
The format of the output tensor is NCHW with C=2; [batch, 2, height, width].
The first channel is the x (width) component.

## warp2d

```python
warp2d(input, displacements, normalized=False, border_mode='clamp', border_value=0.0)
```

Warps the input with the given displacement vector field.



*There is no corresponding gradient op*

#### Args

* ```input```: Input tensor in the format NCHW with a minimum rank of 2.
For rank 2 tensors C == 1 is assumed.
For rank 3 tensors N == 1 is assumed.

* ```displacements```: The tensor storing the displacement vector field.
The format is NCHW with C=2 and the rank is at least 3.
The first channel is the displacement in x direction (width).
The second channel is the displacement in y direction (height).

* ```normalized```: If true then the displacement vectors are normalized with the width and height of the input.

* ```border_mode```: Defines how to handle values outside of the image.
'clamp': Coordinates will be clamped to the valid range.
'value' : Uses 'border_value' outside the image borders.

* ```border_value```: The value used outside the image borders.



#### Returns


The warped input tensor.

