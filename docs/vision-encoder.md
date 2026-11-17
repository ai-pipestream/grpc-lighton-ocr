# The vision encoder on an OpenVINO device

On `GRPC_LIGHTON_DEVICE=openvino:*` the vision encoder is not llama.cpp's
ggml graph. It is an OpenVINO model, exported once from the same projector
file, and the server runs it through the OpenVINO runtime's C++ API on the
configured device. The text model stays on llama.cpp's ggml OpenVINO backend.
The CPU engine (`GRPC_LIGHTON_DEVICE=cpu`) is unchanged and remains the
reference.

## Why

At the pinned llama.cpp revision the ggml OpenVINO backend computes this
encoder wrongly and cannot take a second page size
(`docs/upstream/ggml-openvino-vision.md`; both are fixed by carried patches).
Independent of those faults, an encoder that is one fixed graph is better
served as a model the runtime compiles directly: the plugin's own graph
optimizations and kernels apply, each page size is compiled once and kept,
and the placement can be read back from the runtime.

## Parts

| Part | Where | When it runs |
|---|---|---|
| Graph builder: weights to `ov::Model` | `src/vision/encoder_model.cpp` | export only |
| Projector file reader | `src/vision/gguf_weights.cpp` | export only |
| Export tool | `tools/export_vision_encoder.cpp` | model preparation, from the build stage image |
| Runtime: compile per page size, run, check placement | `src/vision/openvino_encoder.cpp` | in the server |
| Parity tool against the ggml CPU encoder | `tools/vision_encoder_parity.cpp` | qualification |

No Python is involved. The export reads `vision-f16.gguf`, the projector file
`tools/prepare_model.py` already produces, and writes
`vision-encoder-f16.xml` and `vision-encoder-f16.bin`. The server loads those
two files next to the GGUF files and verifies their SHA256 like the others
(`models/SHA256SUMS`). Exporting twice gives identical files.

```sh
docker build -f Dockerfile.openvino --target build -t grpc-lighton-ocr:openvino-build .
scripts/export-vision-encoder.sh grpc-lighton-ocr:openvino-build /path/to/models
```

## The model

```
pixel_values  f32 [1, 3, H, W]    RGB, (value / 255 - 0.5) / 0.5; H, W multiples of 32
image_embeds  f32 [H/32 * W/32, 1024]
```

It follows the graph llama.cpp builds for this projector type
(`tools/mtmd/models/qwen3vl.cpp`), which is what the CPU engine runs:

1. Patch embedding: the checkpoint's two frame kernels, added into one 16x16
   convolution (a still image is both frames), plus bias.
2. The learned 48x48 position table, resized to the page's patch grid
   (bilinear, aligned corners), added.
3. Patches reordered so that the 2x2 patches of one output token are
   adjacent.
4. Twelve pre-norm transformer blocks: fused query/key/value projection,
   two-dimensional rotary positions on query and key (first quarter of a
   head's dimensions turns with the patch row, second with the column, each
   paired with the dimension half a head further on), attention, tanh GELU
   MLP.
5. Final layer norm, the four patches of a token joined, and the two-layer
   projector to the text model's width.

The page size is dynamic in the file. Everything that depends on it, the
resized position table and the rotary tables, is computed inside the graph
from the input's shape. The server reshapes a private copy to the page's
fixed size before compiling, and those subgraphs fold to constants.

Weights keep the precision of the projector file: half-precision tensors are
half-precision constants followed by a conversion, float tensors (biases,
norms, the position table) stay float. The summed patch kernel is stored as
float. The file holds the checkpoint's values exactly.

## What mtmd still does

mtmd loads the projector file into host memory as before and does everything
but encode: it decodes the image, resizes and normalizes it, and lays out the
prompt with the image's token positions. The server takes the preprocessed
pixels from the image chunk (`mtmd_image_tokens_get_pixels`, carried patch
0001), runs the encoder, and gives the embeddings to
`mtmd_helper_decode_image_chunk`. Preprocessing is therefore the CPU engine's,
bit for bit; no second implementation of the resize exists.

mtmd's own encoder is host code and must not run on an accelerator. Its
scheduler callback counts the nodes it computes, and a count other than zero
fails the page.

## Placement

- The model is compiled for the one configured device by name. The server
  never asks for `AUTO`, `HETERO` or `MULTI` and the image has no CPU plugin.
- After compiling a page size the server reads the runtime's
  `EXECUTION_DEVICES` and requires exactly that device.
- On a GPU it walks the runtime model and fails if any executed node has a
  host implementation (the plugin names those `<primitive>_cpu_impl`).
- The startup reference page runs through this encoder and the decoder; a
  failure of either check, or a page that is not read correctly, stops
  startup. The same checks run for every new page size at request time, and a
  failure there is `PAGE_FAILURE_CODE_INFERENCE`.

The startup log carries the receipt, for example:

```
startup placement probe: vision: 992x288: 370 nodes on GPU.0, 0 nodes computed by the host projector | decoder: ...
```

## Page sizes

mtmd resizes a page to multiples of 32 pixels and at most 2048 tokens, so the
set of sizes is finite but not small. Each distinct size is compiled once and
kept, up to six at a time; the least recently used goes first. A page of a new
size pays the compilation inside its own deadline. Set
`GGML_OPENVINO_CACHE_DIR` to a writable directory to keep compiled kernels
across restarts; the encoder uses the same directory as the text model's
backend.

## Measurements

Host: Ryzen 9 9950X with an Arc Pro B70 (xe driver), OpenVINO 2026.4.1,
compute runtime 26.05 from Ubuntu 26.04. Reference: the ggml CPU encoder
(`libmtmd`, flash attention as it selects) with the same F16 weights, fed the
same preprocessed pixels. Tool: `tools/vision_encoder_parity.cpp`. Pages: the
two invoice fixtures of this repository and six pages from gRParse's test
corpus (a report page, a table image, and first pages of a form, a two-column
text, a scan and a long text, rendered at 110 dpi).

"Worst token" is the lowest cosine of any single token's embedding; "low" is
the number of tokens below 0.99. "Encode" is one page with the size already
compiled; "compile" is the first page of a size, without a kernel cache.

### GPU, f32 (what the server uses)

| Page | Size | Tokens | Cosine | Worst token | Low | Encode | Compile |
|---|---|---|---|---|---|---|---|
| invoice | 992x288 | 279 | 0.999996 | 0.999776 | 0 | 41 ms | 6.3 s |
| invoice, tall | 992x640 | 620 | 0.999985 | 0.998010 | 0 | 81 ms | 8.8 s |
| table image | 704x320 | 220 | 0.999997 | 0.999827 | 0 | 21 ms | 5.3 s |
| report page | 1216x1696 | 2014 | 0.999995 | 0.995754 | 0 | 825 ms | 20.8 s |
| form | 928x1216 | 1102 | 0.999996 | 0.998992 | 0 | 302 ms | 12.8 s |
| two-column | 928x1216 | 1102 | 0.999991 | 0.990812 | 0 | 289 ms | kept |
| scan | 928x1216 | 1102 | 0.999973 | 0.996743 | 0 | 293 ms | kept |
| long text | 928x1216 | 1102 | 0.999993 | 0.998730 | 0 | 292 ms | kept |

The OpenVINO CPU device in f32 gives the same agreement to the fifth decimal
(encode 69 ms to 3.7 s). The compiled model has 406 nodes on `GPU.0` and none
with a host implementation.

### GPU, f16 (measured, not used)

| Page | Cosine | Worst token | Low | Encode | Compile |
|---|---|---|---|---|---|
| invoice | 0.999960 | 0.996746 | 0 | 6 ms | 4.2 s |
| invoice, tall | 0.999753 | 0.979880 | 4 | 16 ms | 6.9 s |
| table image | 0.999988 | 0.998962 | 0 | 5 ms | 3.5 s |
| report page | 0.999863 | 0.863030 | 6 | 73 ms | 17.8 s |
| form | 0.999871 | 0.863161 | 5 | 31 ms | 10.4 s |
| two-column | 0.999898 | 0.954212 | 4 | 32 ms | kept |
| scan | 0.999676 | 0.981408 | 3 | 31 ms | kept |
| long text | 0.999694 | 0.806667 | 5 | 31 ms | kept |

Half precision is about ten times faster and moves a handful of tokens per
large page a long way. Those are tokens with very large activations (the last
block's output reaches magnitudes in the thousands), where half precision
runs out of digits. An earlier export that stored the float tensors (norm
parameters, biases, position table) in half precision showed the same pattern
even in f32 arithmetic (long text: worst token 0.807), which is why the export
keeps them float. The encoder is a small part of a page's time, so the server
computes in f32.

### How exact the reference is

The ggml CPU backend rounds activations to the weights' precision in every
matrix product, so its result depends on the weight format:

| Comparison (both ggml CPU) | Page | Cosine | Worst token | Low |
|---|---|---|---|---|
| BF16 weights (the CPU engine) vs F16 weights | invoice | 0.999847 | 0.983428 | 2 |
| | report page | 0.999831 | 0.930431 | 7 |
| | long text | 0.999668 | 0.920952 | 10 |
| F16 weights, flash attention on vs off | report page | 0.999996 | 0.998948 | 0 |
| | long text | 0.999992 | 0.993518 | 0 |

In f32 the exported encoder is closer to the F16 reference than the CPU
engine's own BF16 encoder is.

## End to end

The OpenVINO image on the B70, `ParseDocument` with greedy sampling and a
1024-token cap, compared with the CPU engine (ggml CPU backend, BF16 weights)
on the same eight pages. Three passes: the same text in every pass, no
placement violation, a clean exit, no xe fault in the kernel log.

| Page | Image tokens | Text against the CPU engine | Time a page |
|---|---|---|---|
| invoice | 279 | identical | 6 s |
| invoice, tall | 620 | identical | 6 s |
| table image | 220 | identical | 7.5 s |
| form | 1102 | 1 of 122 words: a ruled line two characters shorter | 11 s |
| two-column | 1102 | 10 of 655 words: box coordinates by up to 4 units, one emphasis mark (both stop at the token cap) | 14 s |
| scan | 1102 | identical (both return an empty reply) | 6 s |
| long text | 1102 | 2 of 992 words: box coordinates by up to 3 units (both stop at the token cap) | 14 s |
| report page | 2014 | identical | 12 s |

GPU memory held by the server: 3.4 GiB after startup, 15.7 GiB after the
first pass, and the same after the second and third.

On the OpenVINO CPU device the two invoices, the table image and the form
are identical to the CPU engine; an earlier build was also identical on the
report page. `scripts/test-openvino-gpu.sh` passes on the B70.

### What the text model's backend forces on the GPU

The page times above are mostly compilation, and they are that because of two
faults of the ggml OpenVINO backend on the GPU that showed once pages of more
than one size and more than 620 tokens went through it. Neither is in the
vision encoder; both are described for the backend's maintainers in
`docs/upstream/ggml-openvino-vision.md`, section 3.

1. **A page's embeddings in several batches.** With the CPU engine's batch of
   512, a 1102-token page is prefilled in three batches, and on the GPU the
   reading then lacked the top of the page: the form came back without its
   heading, first paragraph and one table row. It is the text model, not the
   encoder: the same happened with mtmd's CPU encoder feeding the GPU text
   model, and the exported encoder with one batch gave the CPU engine's text.
   On an accelerator the server therefore prefills a page as one batch.
2. **Reusing compiled graphs across page sizes.** With one batch, a server
   that kept the text model's compiled graphs from page to page crashed with
   a double free on the first page of a second size, whether it kept them
   through one decoder context for all pages or through the backend's
   process-wide cache. With batches of 512 the second way did not crash, was
   fast (0.6 s for the invoice, 5 to 9 s for the large pages) and filled the
   card: the backend compiles a graph per prompt length and GPU memory grew
   with each, to 32 GiB after 16 pages. The server therefore uses a decoder
   context per page, as the CPU engine does. The backend frees its compiled
   graphs with its last context, so every page compiles them again.

What this leaves: correct text at 6 to 14 s a page, where the decode itself
takes 0.4 to 5 s. Getting the rest back needs the backend to take embeddings
of any length in one compiled graph and to survive reuse; that is work on the
backend, not on this server.
