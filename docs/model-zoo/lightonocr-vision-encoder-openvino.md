---
license: apache-2.0
base_model: lightonai/LightOnOCR-3-0.8B
library_name: openvino
pipeline_tag: image-feature-extraction
tags: [ocr, document-understanding, vision-encoder, openvino, intel-gpu]
---

# LightOnOCR-3-0.8B vision encoder, OpenVINO IR

Draft model card for the model zoo (gRParse issue 334). Not published yet.

The vision tower and projector of
[lightonai/LightOnOCR-3-0.8B](https://huggingface.co/lightonai/LightOnOCR-3-0.8B)
(revision `36d8636925327c671bd8e479c517a004f7566bd9`, Apache-2.0) as one
OpenVINO model. It turns a preprocessed page into the image embeddings the
checkpoint's text model takes in place of its image tokens. The text model is
not part of this repository.

It was built to run the encoder on Intel GPUs, where it is qualified on an
Arc Pro B70 with OpenVINO 2026.4.1.

## Files

| File | SHA256 | Size |
|---|---|---|
| `vision-encoder-f16.xml` | `069c37e4f2fc4ccbb611333f8cc089a148e5419bc169de913bae86313267a71d` | 474518 |
| `vision-encoder-f16.bin` | `39fe3a5080be77c576263344ffd66bc9d7ead94ba31195cd6c4f0a4d816cd14c` | 204977560 |

Weights keep the precision of the F16 GGUF conversion they come from: matrix
weights are half-precision constants followed by a conversion to float;
biases, layer norm parameters and the position table are float. No further
compression was applied, so the file holds those values exactly.

## Interface

```
input   pixel_values  f32 [1, 3, H, W]   RGB, (value / 255 - 0.5) / 0.5
output  image_embeds  f32 [(H / 32) * (W / 32), 1024]
```

- `H` and `W` are dynamic and must be multiples of 32. Reshape the model to
  the page's fixed size before compiling; the position table resize and the
  rotary tables inside the graph then fold to constants. A dynamic compile is
  possible but was not measured.
- Resize the page as the checkpoint's processor does (aspect-preserving, both
  sides rounded to multiples of 32, bicubic) before normalizing. The model
  does not resize.
- Output rows are in reading order of the 32-pixel token grid, row by row:
  row `r`, column `c` is at index `r * (W / 32) + c`. Feed them to the text
  model where its image pad tokens are, with the positions the checkpoint
  uses for an image of that grid.
- The model's `rt_info` carries `lighton/token_stride` (32), `patch_size`
  (16), `merge_size` (2), `pixel_format` and `source_file`.

```cpp
ov::Core core;
auto model = core.read_model("vision-encoder-f16.xml");
model->reshape(ov::PartialShape{1, 3, height, width});
auto compiled = core.compile_model(model, "GPU", ov::hint::inference_precision(ov::element::f32));
auto request = compiled.create_infer_request();
// fill request.get_input_tensor() with the normalized page, plane by plane
request.infer();
const ov::Tensor embeddings = request.get_output_tensor();
```

## How it was made

1. The checkpoint's `model.safetensors` (SHA256
   `ae46aa6fd7136d57297ac006ed9ec345b7a2083feb2e99501bedcb014ca3efdf`) was
   converted to a projector GGUF with llama.cpp's `convert_hf_to_gguf.py
   --mmproj --outtype f16` at llama.cpp
   `f3586e66bc916d672773d4a6505513ba91a3595c` (`vision-f16.gguf`, SHA256
   `984817f5c138c8400a171770175ae016e77a904252d3d456dcca65f217874f6b`).
2. `export_vision_encoder` from
   [grpc-lighton-ocr](https://github.com/ai-pipestream/grpc-lighton-ocr)
   read that file and built the graph with the OpenVINO C++ API, op by op,
   following the graph llama.cpp builds for this projector type
   (`tools/mtmd/models/qwen3vl.cpp`). No tracing and no Python.
3. `ov::save_model` wrote the IR with OpenVINO 2026.4.1. The export is
   reproducible: the same inputs give the same two files.

The graph: one 16x16 patch convolution (the checkpoint's two frame kernels
added, a still image being both frames), the learned 48x48 position table
resized to the page's patch grid (bilinear, aligned corners), twelve pre-norm
transformer blocks with two-dimensional rotary positions on query and key and
a tanh GELU MLP, a final layer norm, the four patches of each token joined,
and a two-layer projector. It has no deepstack branch (the checkpoint has
none) and takes still images only.

## Accuracy

Reference: llama.cpp's own CPU encoder (`libmtmd`, ggml CPU backend) with the
same F16 weights and the same preprocessed pixels. Eight document pages, 220
to 2014 tokens each.

| Device and arithmetic | Page cosine | Worst single token | Tokens below 0.99 | Encode (size compiled) | First page of a size |
|---|---|---|---|---|---|
| Arc Pro B70, f32 | 0.99997 to 0.999997 | 0.9908 to 0.9998 | none | 21 to 825 ms | 5 to 21 s |
| Arc Pro B70, f16 | 0.99968 to 0.99999 | 0.807 to 0.999 | 0 to 6 per page | 5 to 73 ms | 3.5 to 18 s |
| CPU plugin (Ryzen 9 9950X), f32 | as B70 f32 | as B70 f32 | none | 69 ms to 3.7 s | 1.6 to 13 s |

For scale: the reference's own result moves by more than the f32 rows when
its weights are BF16 instead of F16 (page cosine 0.9997 to 0.9998, worst token
0.92 to 0.98), because the ggml CPU backend rounds activations to the weight
precision in every matrix product. Use f32 arithmetic when the embeddings must
track the reference; f16 is about ten times faster and moves a few tokens with
very large activations on large pages.

On the B70 the compiled model runs entirely on the GPU (406 nodes in f32,
370 in f16, none with a host implementation). State the arithmetic when
compiling: left alone, the GPU plugin computes in f16 and the CPU plugin in
bfloat16 on processors that have it.

Per-page figures and the method: `docs/vision-encoder.md` in grpc-lighton-ocr.

## Licence

Apache-2.0, as the checkpoint it is derived from. LightOn's licence and
attribution apply to the weights.
