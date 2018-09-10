//
//  lmbspecialops - a collection of tensorflow ops
//  Copyright (C) 2017  Albert Ludwigs University of Freiburg, Pattern Recognition and Image Processing, Computer Vision Group
//  Author(s): Lukas Voegtle <voegtlel@tf.uni-freiburg.de>
//  
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//  
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//  
//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
#include "config.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/platform/cpu_info.h"

#include <webp/encode.h>

using namespace tensorflow;

REGISTER_OP("EncodeWebp")
    .Input("image: uint8")

    .Attr("lossless: bool = false")

    .Attr("lossless_level: int = 6")

    .Attr("preset: {'default', 'picture', 'photo', 'drawing', 'icon', 'text'} = 'default'")
    .Attr("preset_quality: float = 75")

    .Attr("quality: float = -1")
    .Attr("method: int = -1")
    .Attr("image_hint: {'default', 'picture', 'photo', 'graph', 'unspecified'} = 'unspecified'")

    .Attr("target_size: int = -1")
    .Attr("target_PSNR: float = 0")
    .Attr("segments: int = -1")
    .Attr("sns_strength: int = -1")
    .Attr("filter_strength: int = -1")
    .Attr("filter_sharpness: int = -1")
    .Attr("filter_type: int = -1")
    .Attr("autofilter: int = -1")
    .Attr("alpha_compression: int = -1")
    .Attr("alpha_filtering: int = -1")
    .Attr("alpha_quality: int = -1")
    .Attr("pass_: int = -1")

    .Attr("show_compressed: bool = false")
    .Attr("preprocessing: int = -1")
    .Attr("partitions: int = 0")
    .Attr("partition_limit: int = -1")
    .Attr("emulate_jpeg_size: bool = false")
    .Attr("thread_level: bool = false")
    .Attr("low_memory: bool = false")

    .Attr("near_lossless: int = 100")
    .Attr("exact: bool = false")

    .Output("contents: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      using namespace ::tensorflow::shape_inference;
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &unused));
      c->set_output(0, c->Scalar());
      return Status::OK();
    })
    .Doc(R"doc(
Encode uint8 image data as WEBP file.
image: uint8 tensor with shape `[height, width, {3|4}]`.
contents: 0-D.  The WEBP-encoded data.

lossless: bool, Lossless (or lossy) encoding. Default is false.

lossless_level: int, Activate the lossless compression mode with the desired efficiency level between 0 (fastest,
  lowest compression) and 9 (slower, best compression). A good default level is '6', providing a fair tradeoff between
  compression speed and final compressed size.

preset: int, used to initialize the encoder settings:
  'default': Default preset.
  'picture': Picture: Digital picture, like portrait, inner shot
  'photo': Photo: Outdoor photograph, with natural lighting
  'drawing': Drawing: Hand or line drawing, with high-contrast details
  'icon': Icon. Small-sized colorful images
  'text': Text: Text-like
preset_quality: float, between 0 (smallest file) and 100 (biggest), used to initialize the encoder settings.

Encoder configuration:
quality: float, between 0 (smallest file) and 100 (biggest)
method: int, quality/speed trade-off (0=fast, 6=slower-better)

image_hint: int, Hint for image type (lossless only for now):
  'default': Default preset.
  'picture': Digital picture, like portrait, inner shot
  'photo': Outdoor photograph, with natural lighting
  'graph': Discrete tone image (graph, map-tile etc).
  'unspecified': Keep from preset.

Parameters related to lossy compression only:
target_size: int, if non-zero, set the desired target size in bytes. Takes precedence over the 'compression' parameter.
target_PSNR: float, if non-zero, specifies the minimal distortion to try to achieve. Takes precedence over target_size.
segments: int, maximum number of segments to use, in [1..4]
sns_strength: int, Spatial Noise Shaping. 0=off, 100=maximum.
filter_strength: int, range: [0 = off .. 100 = strongest]
filter_sharpness: int, range: [0 = off .. 7 = least sharp]
filter_type: int, filtering type: 0 = simple, 1 = strong (only used if filter_strength > 0 or autofilter > 0)
autofilter: int, Auto adjust filter's strength [0 = off, 1 = on]
alpha_compression: int, Algorithm for encoding the alpha plane (0 = none, 1 = compressed with WebP lossless). Default is 1.
alpha_filtering: int, Predictive filtering method for alpha plane. 0: none, 1: fast, 2: best. Default if 1.
alpha_quality: int, Between 0 (smallest size) and 100 (lossless). Default is 100.
pass_: int, number of entropy-analysis passes (in [1..10]).

show_compressed: bool, if set, export the compressed picture back. In-loop filtering is not applied.
preprocessing: int, preprocessing filter:
  0=none, 1=segment-smooth, 2=pseudo-random dithering
partitions: int, log2(number of token partitions) in [0..3]. Default is set to 0 for easier progressive decoding.
partition_limit: int, quality degradation allowed to fit the 512k limit on prediction modes coding (0: no degradation,
  100: maximum possible degradation).
emulate_jpeg_size: bool, If true, compression parameters will be remapped to better match the expected output size from
  JPEG compression. Generally, the output size will be similar but the degradation will be lower.
thread_level: bool, If set, try and use multi-threaded encoding.
low_memory: bool, If set, reduce memory usage (but increase CPU use).

near_lossless: int, Near lossless encoding [0 = max loss .. 100 = off (default)].
exact: bool, if set, preserve the exact RGB values under transparent area. Otherwise, discard this invisible RGB
  information for better compression. The default value is false.

)doc");




// Encode the contents to a WEBP file
class EncodeWebpOp : public OpKernel {
 public:
  explicit EncodeWebpOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("lossless", &lossless_));

    OP_REQUIRES_OK(context, context->GetAttr("lossless_level", &lossless_level_));

    std::string preset;
    OP_REQUIRES_OK(context, context->GetAttr("preset", &preset));
    if (preset == "default") {
      preset_ = WEBP_PRESET_DEFAULT;
    } else if (preset == "picture") {
      preset_ = WEBP_PRESET_PICTURE;
    } else if (preset == "photo") {
      preset_ = WEBP_PRESET_PHOTO;
    } else if (preset == "drawing") {
      preset_ = WEBP_PRESET_DRAWING;
    } else if (preset == "icon") {
      preset_ = WEBP_PRESET_ICON;
    } else if (preset == "text") {
      preset_ = WEBP_PRESET_TEXT;
    } else {
      OP_REQUIRES(context, false, errors::InvalidArgument("Invalid preset: ", preset));
    }

    OP_REQUIRES_OK(context, context->GetAttr("preset_quality", &preset_quality_));

    OP_REQUIRES_OK(context, context->GetAttr("quality", &quality_));
    OP_REQUIRES_OK(context, context->GetAttr("method", &method_));
    std::string image_hint;
    OP_REQUIRES_OK(context, context->GetAttr("image_hint", &image_hint));
    if (image_hint == "default") {
      image_hint_ = WEBP_HINT_DEFAULT;
    } else if (image_hint == "picture") {
      image_hint_ = WEBP_HINT_PICTURE;
    } else if (image_hint == "photo") {
      image_hint_ = WEBP_HINT_PHOTO;
    } else if (image_hint == "graph") {
      image_hint_ = WEBP_HINT_GRAPH;
    } else if (image_hint == "unspecified") {
      image_hint_ = WEBP_HINT_LAST;
    } else {
      OP_REQUIRES(context, false, errors::InvalidArgument("Invalid image_hint: ", preset));
    }
        
    OP_REQUIRES_OK(context, context->GetAttr("target_size", &target_size_));
    OP_REQUIRES_OK(context, context->GetAttr("target_PSNR", &target_PSNR_));
    OP_REQUIRES_OK(context, context->GetAttr("segments", &segments_));
    OP_REQUIRES_OK(context, context->GetAttr("sns_strength", &sns_strength_));
    OP_REQUIRES_OK(context, context->GetAttr("filter_strength", &filter_strength_));
    OP_REQUIRES_OK(context, context->GetAttr("filter_sharpness", &filter_sharpness_));
    OP_REQUIRES_OK(context, context->GetAttr("filter_type", &filter_type_));
    OP_REQUIRES_OK(context, context->GetAttr("autofilter", &autofilter_));
    OP_REQUIRES_OK(context, context->GetAttr("alpha_compression", &alpha_compression_));
    OP_REQUIRES_OK(context, context->GetAttr("alpha_filtering", &alpha_filtering_));
    OP_REQUIRES_OK(context, context->GetAttr("alpha_quality", &alpha_quality_));
    OP_REQUIRES_OK(context, context->GetAttr("pass_", &pass_));
        
    OP_REQUIRES_OK(context, context->GetAttr("show_compressed", &show_compressed_));
    OP_REQUIRES_OK(context, context->GetAttr("preprocessing", &preprocessing_));
    OP_REQUIRES_OK(context, context->GetAttr("partitions", &partitions_));
    OP_REQUIRES_OK(context, context->GetAttr("partition_limit", &partition_limit_));
    OP_REQUIRES_OK(context, context->GetAttr("emulate_jpeg_size", &emulate_jpeg_size_));
    OP_REQUIRES_OK(context, context->GetAttr("thread_level", &thread_level_));
    OP_REQUIRES_OK(context, context->GetAttr("low_memory", &low_memory_));
        
    OP_REQUIRES_OK(context, context->GetAttr("near_lossless", &near_lossless_));
    OP_REQUIRES_OK(context, context->GetAttr("exact", &exact_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& image = context->input(0);
    OP_REQUIRES(context, image.dims() == 3,
                errors::InvalidArgument("data must be 3-dimensional",
                                        image.shape().DebugString()));
    const uint32 height = static_cast<uint32>(image.dim_size(0));
    const uint32 width = static_cast<uint32>(image.dim_size(1));
    const uint32 channels = static_cast<uint32>(image.dim_size(2));

    OP_REQUIRES(context, channels == 3 || channels == 4,
                        errors::InvalidArgument("data must be [height, width, {3|4}]-shaped",
                                                image.shape().DebugString()));
    OP_REQUIRES(context, height < WEBP_MAX_DIMENSION && width < WEBP_MAX_DIMENSION,
                            errors::InvalidArgument("width or height too large: ",
                                                    image.shape().DebugString(), " > ", WEBP_MAX_DIMENSION));

    // Generate encoder configuration
    WebPConfig config;
    OP_REQUIRES(context, WebPConfigPreset(&config, preset_, preset_quality_),
                                  errors::InvalidArgument("Invalid preset - quality data"));
    if (lossless_) {
      OP_REQUIRES(context, WebPConfigLosslessPreset(&config, lossless_level_),
                                        errors::InvalidArgument("Invalid lossless quality"));
    }

    if (quality_ != -1) config.quality = quality_;

    if (method_ != -1) config.method = method_;
    if (image_hint_ != WEBP_HINT_LAST) config.image_hint = image_hint_;

    if (target_size_ != -1) config.target_size = target_size_;
    if (target_PSNR_ != 0) config.target_PSNR = target_PSNR_;
    if (segments_ != -1) config.segments = segments_;
    if (sns_strength_ != -1) config.sns_strength = sns_strength_;
    if (filter_strength_ != -1) config.filter_strength = filter_strength_;
    if (filter_sharpness_ != -1) config.filter_sharpness = filter_sharpness_;
    if (filter_type_ != -1) config.filter_type = filter_type_;
    if (autofilter_ != -1) config.autofilter = autofilter_;
    if (alpha_compression_ != -1) config.alpha_compression = alpha_compression_;
    if (alpha_filtering_ != -1) config.alpha_filtering = alpha_filtering_;
    if (alpha_quality_ != -1) config.alpha_quality = alpha_quality_;
    if (pass_ != -1) config.pass = pass_;

    config.show_compressed = show_compressed_;
    if (preprocessing_ != -1) config.preprocessing = preprocessing_;
    if (partitions_ != -1) config.partitions = partitions_;
    if (partition_limit_ != -1) config.partition_limit = partition_limit_;
    config.emulate_jpeg_size = emulate_jpeg_size_;
    config.thread_level = thread_level_;
    config.low_memory = low_memory_;

    if (near_lossless_ != -1) config.near_lossless = near_lossless_;
    config.exact = exact_;

    // Validate config
    OP_REQUIRES(context, WebPValidateConfig(&config),
                                      errors::InvalidArgument("Invalid encoder configuration"));

    // Create encoder
    WebPPicture pic;
    OP_REQUIRES(context, WebPPictureInit(&pic), errors::InvalidArgument("WEBP internal error"));
    pic.use_argb = 1;
    pic.width = width;
    pic.height = height;
    OP_REQUIRES(context, WebPPictureAlloc(&pic), errors::InvalidArgument("WEBP internal error"));

    if (channels == 3) {
      OP_REQUIRES(context, WebPPictureImportRGB(&pic, image.flat<uint8>().data(), width * channels), errors::InvalidArgument("WEBP internal error"));
    } else {
      OP_REQUIRES(context, WebPPictureImportRGBA(&pic, image.flat<uint8>().data(), width * channels), errors::InvalidArgument("WEBP internal error"));
    }

    // Set up a byte-writing method (write-to-memory)
    WebPMemoryWriter writer;
    WebPMemoryWriterInit(&writer);
    pic.writer = WebPMemoryWrite;
    pic.custom_ptr = &writer;

    // Encode
    int ok = WebPEncode(&config, &pic);
    // Free input
    WebPPictureFree(&pic);

    // Check for error
    if (!ok) {
      WebPMemoryWriterClear(&writer);
    }
    OP_REQUIRES(context, ok, errors::InvalidArgument("WEBP encoding error: ", pic.error_code));

    // Create output
    Tensor* output = NULL;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({}), &output));

    // Store output data
    std::string* output_data = &output->scalar<string>()();
    output_data->reserve(writer.size);
    output_data->append((const char*)(writer.mem), writer.size);

    // Clear writer
    WebPMemoryWriterClear(&writer);
  }

private:
  bool lossless_;

  int lossless_level_;

  WebPPreset preset_;
  float preset_quality_;

  float quality_;
  int method_;
  WebPImageHint image_hint_;

  int target_size_;
  float target_PSNR_;
  int segments_;
  int sns_strength_;
  int filter_strength_;
  int filter_sharpness_;
  int filter_type_;
  int autofilter_;
  int alpha_compression_;
  int alpha_filtering_;
  int alpha_quality_;
  int pass_;

  bool show_compressed_;
  int preprocessing_;
  int partitions_;
  int partition_limit_;
  bool emulate_jpeg_size_;
  bool thread_level_;
  bool low_memory_;

  int near_lossless_;
  bool exact_;
};
REGISTER_KERNEL_BUILDER(Name("EncodeWebp").Device(DEVICE_CPU), EncodeWebpOp);

