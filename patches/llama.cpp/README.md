# Carried llama.cpp patches

`cmake/PatchLlama.cmake` applies every `*.patch` here, in name order, to the
llama.cpp revision pinned in `CMakeLists.txt`
(`f3586e66bc916d672773d4a6505513ba91a3595c`). The step first restores that
revision, so the fetched tree always equals pin plus these patches. A patch
that no longer applies stops the configure.

Each patch is a plain `git diff` against the pinned revision and stands alone.
All three also apply unchanged to llama.cpp `master` at `d3c3e24`
(2026-11-16).

| Patch | Area | Purpose | Upstream |
|---|---|---|---|
| `0001-mtmd-preprocessed-pixels.patch` | `tools/mtmd` | `mtmd_image_tokens_get_pixels()`: the page exactly as mtmd resized and normalized it. Lets the server run the vision encoder through the OpenVINO runtime and hand the result to `mtmd_helper_decode_image_chunk()`. | Ours; small enough to offer, not required by anyone else yet. |
| `0002-ggml-openvino-permute-view.patch` | `ggml/src/ggml-openvino` | A `PERMUTE` of a `VIEW` was translated correctly only for the axis order `(0, 2, 1, 3)`. A vision encoder without flash attention permutes its value `VIEW` with `(1, 2, 0, 3)` and got scrambled values. | For the backend's maintainers: `docs/upstream/ggml-openvino-vision.md`. |
| `0003-ggml-openvino-fixed-input-shapes.patch` | `ggml/src/ggml-openvino` | The backend's per-context graph cache is keyed without shapes, so a second image size was served from the graph compiled for the first one and failed. A cached graph is now reused only when its fixed-shape inputs still have the same shape. | Same write-up. grpc-asr carries a fix for one case of the same cache (whisper's flat KV buffer); this is the general form. |

The server does not depend on 0002 and 0003 for OCR: on an OpenVINO device its
vision encoder is the exported OpenVINO model (`src/vision`), not the ggml
graph. They are carried because they are small, tested here
(`tests/openvino_backend_test.cpp`, in every OpenVINO build) and meant to go
upstream, and so that the ggml vision path is correct should it be used.

After a pin move: regenerate each patch against the new revision, or drop it
when upstream has the change, and keep `docs/upstream` in step.
