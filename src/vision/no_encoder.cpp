// The vision encoder of a build without OpenVINO (LIGHTON_OPENVINO=OFF):
// there is none. The CPU engine never constructs one; device selection
// refuses an OpenVINO device in such a build before this could be reached.
#include <stdexcept>

#include "vision/openvino_encoder.h"

namespace lighton::vision {

struct OpenvinoEncoder::Impl {
  EncoderPlacement placement;
};

std::string EncoderPlacement::summary() const { return "no OpenVINO in this build"; }

OpenvinoEncoder::OpenvinoEncoder(const EncoderOptions&) {
  throw std::runtime_error("this build has no OpenVINO vision encoder (LIGHTON_OPENVINO is OFF)");
}
OpenvinoEncoder::~OpenvinoEncoder() = default;
uint32_t OpenvinoEncoder::token_stride() const { return 0; }
uint32_t OpenvinoEncoder::out_hidden() const { return 0; }
uint64_t OpenvinoEncoder::compilations() const { return 0; }
const EncoderPlacement& OpenvinoEncoder::placement() const { return impl_->placement; }
std::vector<float> OpenvinoEncoder::encode(std::span<const float>, uint32_t, uint32_t) {
  throw std::runtime_error("this build has no OpenVINO vision encoder (LIGHTON_OPENVINO is OFF)");
}

}  // namespace lighton::vision
