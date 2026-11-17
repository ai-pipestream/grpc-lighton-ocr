# ggml OpenVINO backend: two faults on a vision encoder graph

For the maintainers of `ggml/src/ggml-openvino` in llama.cpp. Found at
llama.cpp `f3586e66bc916d672773d4a6505513ba91a3595c` with OpenVINO 2026.4.1,
running the vision encoder of a Qwen3-VL style projector
(`clip.projector_type = qwen3vl_merger`, 12 layers, F16) through `libmtmd`
with the OpenVINO backend as the projector's device. The two patches are
`patches/llama.cpp/0002-ggml-openvino-permute-view.patch` and
`0003-ggml-openvino-fixed-input-shapes.patch` in this repository; each is a
plain `git diff` against that revision. Both also apply unchanged to
llama.cpp `master` at `d3c3e24` (2026-11-16), where the code they change is
the same. Nothing here has been sent upstream yet.

Test stack: OpenVINO 2026.4.1 (Intel's release archive), Ubuntu 26.04, NEO
compute runtime 26.05; an Intel Arc Pro B70 on the xe driver and, as the
OpenVINO CPU device, an AMD Ryzen 9 9950X. llama.cpp's `test-backend-ops` was
not run against these patches.

The text model of the same checkpoint runs on the backend unchanged and
matches the CPU backend. Both faults are in the vision path only.

## 1. `PERMUTE` of a `VIEW` assumes the axis order (0, 2, 1, 3)

### Symptom

The encoder output is wrong: an invoice page is read as a run of unrelated
digits. Every node is placed on the OpenVINO device, in one split, and
nothing is reported.

### Where

`mtmd` falls back to the non-flash attention path on this backend
(`warmup: flash attention not supported by OPENVINO`). There
`clip_graph::build_attn()` takes the value tensor, a `VIEW` into the fused
query/key/value projection, and permutes it directly:

```cpp
ggml_tensor * v = ggml_permute(ctx0, v_cur, 1, 2, 0, 3);
v = ggml_cont(ctx0, v);
```

`GgmlOvDecoder::compute_op_case()` gives a `PERMUTE` whose source is a `VIEW`
of a computed tensor op case 2, described as "rope'ed query tensor".
`translate_permute()` then reshapes the source to
`[n_seq, -1, n_heads, head_size]` before the `Transpose`, and reads `n_heads`
and `head_size` from **output** axes 1 and 3:

```cpp
auto n_heads = Constant(output_shape[1]);
auto head_size = Constant(output_shape[3]);
```

That holds for the permutation (0, 2, 1, 3) of an attention query, where the
output is `[n_seq, n_heads, n_tokens, head_size]`. For (1, 2, 0, 3) the output
is `[1, n_heads, head_size, n_tokens]`, so the reshape target becomes
`[1, -1, n_heads, n_tokens]`: the value tensor is reinterpreted, not
transposed, and the `Transpose` that follows works on scrambled data.

### Measurement

The vision graph was cut after node *k* and computed by the ggml CPU backend
and by the OpenVINO backend (OpenVINO CPU device), same weights, same page
(the 992x288 reference invoice, 1116 patches). Cosine between the two:

| Node | Op | Cosine |
|---|---|---|
| 30 `inp_pos_emb` | ADD | 0.999999 |
| 37 `Qcur_rope-0` | ROPE | 0.999996 |
| 39 `Kcur_rope-0` | ROPE | 0.999996 |
| **42 `Vcur-0 (permuted) (cont)`** | PERMUTE + CONT | **0.001** |
| 45 key x query | MUL_MAT | 0.999999 |
| 46 | SOFT_MAX | 0.999784 |
| 47 value x weights | MUL_MAT | 0.006 |
| 61 `layer_out-0` | ADD | 0.769 |

The first node that differs is the permuted value `VIEW` of layer 0. Query and
key go through `ROPE` before their `PERMUTE` and take op case 1, which is why
they are right. (The small differences elsewhere are the OpenVINO CPU plugin
computing in bfloat16 on this processor.)

### Fix

Read `n_heads` and `head_size` from the `VIEW` being restored, which is the
layout the reshape is meant to produce, instead of from the output:

```cpp
auto view_shape = context.get_input_shape(0).to_shape();
auto n_heads = Constant(view_shape[2]);
auto head_size = Constant(view_shape[3]);
```

For (0, 2, 1, 3) these are the same two numbers as before, so the attention
query path of the language models is unchanged.

## 2. A compiled graph is reused for another fixed input shape

### Symptom

The first image size works. A page of any other size fails inside OpenVINO
with an input size mismatch, for example a graph compiled for 448x448 given a
288x992 tensor.

The same cache gives the same kind of failure in whisper.cpp, seen on an Arc
Pro B70 and on the CPU device: `Can't set the input tensor with index: 4,
because the model input (shape=[1,1,1,786432]) and the tensor
(shape=(1.1.1.5505024)) are incompatible`. There the fixed-shape input is
whisper's flat KV buffer, which whisper.cpp replaces with a larger one.
grpc-asr carries a fix for that case
(`docs/upstream/ggml-openvino-flat-kv-reuse.patch` in that repository), which
records the flat KV buffer sizes in `ModelParams` and compares them in
`can_reuse_dynamically()` and `can_reuse_statically()`. The patch here makes
the same decision at the same place for every input that is converted with a
fixed shape, so it is the general form of that fix; the flat KV buffer is one
such input. It was not run with whisper.cpp. The two patches apply together.

### Where

`ov_graph_compute_dynamic()` and `ov_graph_compute_static()` look up the
decoder and the infer request of a graph by `graph_key`: node count, leaf
count and the names of the first and last node. No shape is part of it. That
is intended for the language model, whose token, position, mask and KV
inputs are converted with dynamic dimensions so that one compiled model
serves every batch. An input that is not one of those kinds, such as the
`inp_raw` image of a vision encoder, is converted with its concrete shape
(`get_graph_input_shape()`, last branch). A second image size builds a graph
with the same node count and the same names, finds the cached entry, and binds
its tensor to a model compiled for the first size.

The process-wide compiled-model cache is not affected: its key
(`compiled_graph_key()`) includes every tensor's `ne`.

### Fix

The decoder records, once, the inputs its model was converted with at a fully
static shape, by their position in the graph (node index and source slot; the
graph a backend receives is a scheduler split and lists no leafs) and their
`ne`. On a `graph_key` hit the backend asks
`ggml_decoder->fixed_inputs_match(cgraph)`; a different shape at any of those
positions turns the hit into a miss, and the existing miss branch converts
the graph and replaces the entry. Dynamic inputs are not recorded, so reuse
across token counts is untouched. The check costs one comparison per fixed
input per call; a language model graph has few or none.

Going back to a size seen earlier takes the miss branch again, which then
finds the compiled model in the process-wide cache by its exact key and only
creates a new infer request.

### Not addressed

Every distinct image size leaves a compiled model in the process-wide cache
for the life of the process. A service that sees many sizes will want a
bound there.

## 3. Observed on the text model, not fixed here

Seen while serving pages end to end on the Arc Pro B70 with the text model
(a 24-layer model with gated-delta-net layers, F16) on this backend. None of
these is root-caused; the server works around the first two.

**A page's embeddings in several batches read differently on the GPU.** With
`n_batch = 512`, a page of 1102 image tokens is prefilled as 512 + 512 + 78
embeddings. On the GPU the generated text then lacks the top of the page (the
heading and first paragraph of a form, one table row) and stops earlier (511
tokens instead of 591). With the whole prompt in one batch (`n_batch = 4096`)
the GPU gives the CPU backend's text, with box coordinates off by a few units
at most. The embeddings are not the cause: the result is the same with
llama.cpp's own CPU encoder feeding the GPU text model. The OpenVINO CPU
device gives the CPU backend's text with `n_batch = 512`. Pages of up to 620
tokens (512 + 108) were not affected.

**One compiled model per prompt length, and device memory with each.** The
process-wide cache is keyed by every tensor's shape (`compiled_graph_key`),
so a host that creates a context per request compiles the prefill graph again
for every new number of image tokens. A server that kept them all, with
`n_batch = 512`, held 5.5 GiB of GPU memory after startup, 11.8 GiB after two
page sizes and the card's 32 GiB after 16 pages of eight sizes, which points
at a copy of the weights per compiled graph; that was not looked at directly.
Within one context the backend finds graphs by the shape-free `graph_key`,
the cache fault 2 above is about.

**A crash when compiled graphs are reused after a page of another size.**
With a page's prompt in one batch, a server that reused compiled graphs
across pages crashed with `double free or corruption (!prev)` on the first
page of a second size: the image prefill graph was compiled for the new
number of embeddings, and the crash came in the next graph, the short
text-token graph, which was reused. It happened in three runs that sent one
992x288 page and then a 992x640 page (a run that sent two 992x288 pages
first passed), in two arrangements: one `llama_context`
for all pages (reuse through the context's own cache; the prefill graph should
take patch 3's miss branch there, which was not traced), and a context per
page with a second, idle backend instance keeping the process-wide cache
alive (reuse through `compiled_graph_key`, where patch 3 plays no part). With `n_batch = 512` the
second arrangement served 16 pages of eight sizes without a crash. The
server now lets every page compile its own graphs, which the backend does
when a page's context is its only instance.

## Verification

`tests/openvino_backend_test.cpp` in this repository builds the same attention
block (fused projection, three `VIEW`s, `ROPE` on query and key, value
permuted with (1, 2, 0, 3), no flash attention) and runs it at several token
counts through one scheduler on the OpenVINO CPU device, against the ggml CPU
backend. It needs no model file and no GPU, and runs in every OpenVINO build
of this repository. Without patch 1 it fails at the first size (cosine 0.41
to the CPU backend). A build with fault 1 fixed and fault 2 still present
failed it at the second size, with OpenVINO's incompatible-shape message.

With both patches the real encoder, through `libmtmd` with the OpenVINO
backend as its device, agrees with the ggml CPU backend again (cosine of the
whole embedding, same F16 weights, flash attention off on both sides):

| Page | Tokens | OpenVINO CPU device | Arc Pro B70 |
|---|---|---|---|
| 992x288 invoice | 279 | 0.999098 | 0.999953 |
| 704x320 table image | 220 | 0.999354 | 0.999976 |
| 928x1216 text page | 1102 | 0.997876 | 0.999938 |

Unpatched, the same comparison is meaningless (the first block's output is at
cosine 0.77). The CPU device column is lower because that plugin computes in
bfloat16 on this processor; the GPU computes in half precision. Each page ran
in a process of its own, so these runs show fault 1 fixed on both devices;
fault 2 is covered by the test above on the CPU device only.
