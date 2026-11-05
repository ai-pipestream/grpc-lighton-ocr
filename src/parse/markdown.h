#pragma once

#include <cstdint>
#include <string>

#include "ai/pipestream/document/v1/document.pb.h"

namespace lighton {

namespace docv1 = ai::pipestream::document::v1;

// Turns one page of LightOnOCR-3 grounding markdown into a Document fragment.
// Markers are `![label](x1, y1, x2, y2)` on the 0-1000 grid. A box with area
// becomes provenance in page pixels when width and height are non-zero, and
// stays on that grid when they are zero. An empty reply is an error.
bool parse_lightonocr_markdown(const std::string& content, uint32_t page_no,
                               uint32_t width_px, uint32_t height_px,
                               docv1::Document* out, std::string* error);

}  // namespace lighton
