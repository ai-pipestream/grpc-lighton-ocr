#include "parse/markdown.h"

#include <cctype>
#include <regex>
#include <string>
#include <utility>
#include <vector>

namespace lighton {
namespace {

struct Block {
  std::string label;
  std::string text;
  int x0 = 0;
  int y0 = 0;
  int x1 = 0;
  int y1 = 0;
  bool has_box = false;
};

bool picture_label(const std::string& label) {
  return label == "image" || label == "chart" || label == "header_image" ||
         label == "footer_image";
}

bool furniture_label(const std::string& label) {
  return label == "header" || label == "footer" || label == "page_number" ||
         label == "header_image" || label == "footer_image";
}

std::string strip_preamble(std::string content) {
  static const std::regex think(R"(^\s*(?:<think>[\s\S]*?)?</think>\s*)");
  static const std::regex partial(R"(!\[[A-Za-z_][\w-]*\+?\]\(\s*[\d,\s]*$)");
  content = std::regex_replace(content, think, "");
  content = std::regex_replace(content, partial, "");
  return content;
}

bool fence_delimiter(const std::string& line, std::string* delimiter) {
  size_t i = 0;
  while (i < line.size() && i < 3 && line[i] == ' ') i++;
  if (i >= line.size()) return false;
  const char mark = line[i];
  if (mark != '`' && mark != '~') return false;
  size_t j = i;
  while (j < line.size() && line[j] == mark) j++;
  if (j - i < 3) return false;
  *delimiter = line.substr(i, j - i);
  return true;
}

std::vector<Block> split_blocks(const std::string& content) {
  static const std::regex marker(
      R"(!\[([A-Za-z_][\w-]*)\+?\]\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)[ \t]*)");
  std::vector<Block> blocks;
  Block current;
  current.label = "text";
  std::string fence;
  size_t line_start = 0;
  const std::string text = content + "\n";
  while (line_start <= content.size()) {
    const size_t line_end = text.find('\n', line_start);
    std::string line = text.substr(line_start, line_end - line_start);
    if (!line.empty() && line.back() == '\r') line.pop_back();
    const std::string kept_newline = line_end < text.size() ? "\n" : "";
    std::string rest = line;
    if (fence.empty()) {
      std::smatch match;
      while (std::regex_search(rest, match, marker)) {
        current.text.append(match.prefix().str());
        if (!current.text.empty() || current.has_box || blocks.empty()) {
          blocks.push_back(current);
        }
        current = Block{};
        current.label = match[1].str();
        current.x0 = std::stoi(match[2].str());
        current.y0 = std::stoi(match[3].str());
        current.x1 = std::stoi(match[4].str());
        current.y1 = std::stoi(match[5].str());
        current.has_box = true;
        rest = match.suffix().str();
      }
    }
    std::string delimiter;
    if (fence_delimiter(rest, &delimiter)) {
      if (fence.empty()) {
        fence = delimiter;
      } else if (delimiter[0] == fence[0] && delimiter.size() >= fence.size()) {
        const std::string after = rest.substr(rest.find(delimiter) + delimiter.size());
        if (after.find_first_not_of(" \t") == std::string::npos) fence.clear();
      }
    }
    current.text.append(rest);
    current.text.append(kept_newline);
    if (line_end == std::string::npos || line_end >= content.size()) break;
    line_start = line_end + 1;
  }
  blocks.push_back(std::move(current));
  return blocks;
}

int heading_level(const std::string& text) {
  int level = 0;
  for (char c : text) {
    if (c == '#') {
      level++;
      continue;
    }
    break;
  }
  if (level >= 2 && level < static_cast<int>(text.size()) && text[static_cast<size_t>(level)] == ' ') {
    return level;
  }
  return 0;
}

std::string trim_copy(std::string text) {
  while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front())) != 0) {
    text.erase(text.begin());
  }
  while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back())) != 0) {
    text.pop_back();
  }
  return text;
}

void set_box(docv1::ProvenanceItem* prov, const Block& block, uint32_t width, uint32_t height) {
  const bool on_grid = block.x0 >= 0 && block.y0 >= 0 && block.x0 < block.x1 &&
                       block.y0 < block.y1 && block.x1 <= 1000 && block.y1 <= 1000;
  if (!block.has_box || !on_grid) return;
  docv1::BoundingBox* box = prov->mutable_bbox();
  const double width_d = width == 0 ? 1000.0 : static_cast<double>(width);
  const double height_d = height == 0 ? 1000.0 : static_cast<double>(height);
  box->set_l(block.x0 / 1000.0 * width_d);
  box->set_t(block.y0 / 1000.0 * height_d);
  box->set_r(block.x1 / 1000.0 * width_d);
  box->set_b(block.y1 / 1000.0 * height_d);
  box->set_coord_origin(docv1::COORD_ORIGIN_TOPLEFT);
}

docv1::TextItemBase* add_text(docv1::Document* document, const Block& block, uint32_t page_no,
                              uint32_t width, uint32_t height, docv1::DocItemLabel label,
                              int section_level) {
  const bool furniture = furniture_label(block.label);
  docv1::GroupItem* parent = furniture ? document->mutable_furniture() : document->mutable_body();
  docv1::BaseTextItem* item = document->add_texts();
  docv1::TextItemBase* base = nullptr;
  if (label == docv1::DOC_ITEM_LABEL_TITLE) {
    base = item->mutable_title()->mutable_base();
  } else if (label == docv1::DOC_ITEM_LABEL_SECTION_HEADER || section_level > 0) {
    auto* header = item->mutable_section_header();
    header->set_level(section_level == 0 ? 1 : section_level);
    base = header->mutable_base();
    label = docv1::DOC_ITEM_LABEL_SECTION_HEADER;
  } else if (label == docv1::DOC_ITEM_LABEL_FORMULA) {
    base = item->mutable_formula()->mutable_base();
  } else {
    base = item->mutable_text()->mutable_base();
  }
  const std::string ref = "#/texts/" + std::to_string(document->texts_size() - 1);
  base->set_self_ref(ref);
  base->mutable_parent()->set_ref(furniture ? "#/furniture" : "#/body");
  base->set_content_layer(furniture ? docv1::CONTENT_LAYER_FURNITURE : docv1::CONTENT_LAYER_BODY);
  base->set_label(label);
  const std::string text = trim_copy(block.text);
  base->set_text(text);
  base->set_orig(text);
  docv1::ProvenanceItem* prov = base->add_prov();
  prov->set_page_no(static_cast<int32_t>(page_no));
  set_box(prov, block, width, height);
  parent->add_children()->set_ref(ref);
  return base;
}

void add_picture(docv1::Document* document, const Block& block, uint32_t page_no, uint32_t width,
                 uint32_t height) {
  const bool furniture = furniture_label(block.label);
  docv1::GroupItem* parent = furniture ? document->mutable_furniture() : document->mutable_body();
  docv1::PictureItem* picture = document->add_pictures();
  const std::string ref = "#/pictures/" + std::to_string(document->pictures_size() - 1);
  picture->set_self_ref(ref);
  picture->mutable_parent()->set_ref(furniture ? "#/furniture" : "#/body");
  picture->set_content_layer(furniture ? docv1::CONTENT_LAYER_FURNITURE
                                       : docv1::CONTENT_LAYER_BODY);
  picture->set_label(docv1::DOC_ITEM_LABEL_PICTURE);
  docv1::ProvenanceItem* prov = picture->add_prov();
  prov->set_page_no(static_cast<int32_t>(page_no));
  set_box(prov, block, width, height);
  parent->add_children()->set_ref(ref);
}

docv1::DocItemLabel label_of(const std::string& label) {
  if (label == "title") return docv1::DOC_ITEM_LABEL_TITLE;
  if (label == "section") return docv1::DOC_ITEM_LABEL_SECTION_HEADER;
  if (label == "header") return docv1::DOC_ITEM_LABEL_PAGE_HEADER;
  if (label == "footer" || label == "page_number") return docv1::DOC_ITEM_LABEL_PAGE_FOOTER;
  if (label == "footnote") return docv1::DOC_ITEM_LABEL_FOOTNOTE;
  if (label == "caption") return docv1::DOC_ITEM_LABEL_CAPTION;
  if (label == "formula") return docv1::DOC_ITEM_LABEL_FORMULA;
  if (label == "code") return docv1::DOC_ITEM_LABEL_CODE;
  if (label == "table") return docv1::DOC_ITEM_LABEL_TABLE;
  return docv1::DOC_ITEM_LABEL_TEXT;
}

}  // namespace

bool parse_lightonocr_markdown(const std::string& content, uint32_t page_no, uint32_t width_px,
                               uint32_t height_px, docv1::Document* out, std::string* error) {
  if (out == nullptr || error == nullptr) return false;
  const std::string body = strip_preamble(content);
  if (trim_copy(body).empty()) {
    *error = "empty lightonocr reply";
    return false;
  }
  out->Clear();
  out->set_schema_name("docling_document_v2");
  out->set_version("1.10.0");
  out->set_name("page-" + std::to_string(page_no));
  out->mutable_body()->set_self_ref("#/body");
  out->mutable_body()->set_content_layer(docv1::CONTENT_LAYER_BODY);
  out->mutable_furniture()->set_self_ref("#/furniture");
  out->mutable_furniture()->set_content_layer(docv1::CONTENT_LAYER_FURNITURE);
  docv1::PageItem& page = (*out->mutable_pages())[static_cast<int32_t>(page_no)];
  page.set_page_no(static_cast<int32_t>(page_no));
  if (width_px != 0 && height_px != 0) {
    page.mutable_size()->set_width(width_px);
    page.mutable_size()->set_height(height_px);
  }
  for (const Block& block : split_blocks(body)) {
    const std::string text = trim_copy(block.text);
    if (text.empty() && !picture_label(block.label)) continue;
    if (picture_label(block.label)) {
      add_picture(out, block, page_no, width_px, height_px);
      continue;
    }
    const int hashes = block.label == "title" ? heading_level(text) : 0;
    docv1::DocItemLabel label = label_of(block.label);
    if (hashes >= 2) label = docv1::DOC_ITEM_LABEL_SECTION_HEADER;
    add_text(out, block, page_no, width_px, height_px, label, hashes);
  }
  return true;
}

}  // namespace lighton
