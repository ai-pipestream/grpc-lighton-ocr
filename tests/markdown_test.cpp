#include <cstdlib>
#include <iostream>
#include <string>

#include "ai/pipestream/document/v1/document.pb.h"
#include "parse/markdown.h"

namespace {

int failures = 0;

void expect(bool ok, const std::string& message) {
  if (ok) return;
  std::cerr << "FAIL " << message << '\n';
  failures++;
}

}  // namespace

int main() {
  lighton::docv1::Document document;
  std::string error;
  const std::string reply =
      "![title](10, 20, 400, 80)\n"
      "# Introduction\n"
      "![text](10, 100, 900, 400)\n"
      "End of the left column\n"
      "![footer](10, 960, 200, 990)\n"
      "1\n";
  expect(lighton::parse_lightonocr_markdown(reply, 1, 1000, 1000, &document, &error), error);
  expect(document.version() == "1.10.0", "schema stamp");
  expect(document.texts_size() == 3, "title, text, footer");
  expect(document.texts(0).has_title(), "first block is a title");
  expect(document.texts(0).title().base().text() == "# Introduction", "title text");
  const auto& box = document.texts(1).text().base().prov(0).bbox();
  expect(box.l() == 10 && box.t() == 100 && box.r() == 900 && box.b() == 400, "text box scaled");
  expect(document.texts(2).text().base().content_layer() ==
             lighton::docv1::CONTENT_LAYER_FURNITURE,
         "footer is furniture");
  expect(document.furniture().children_size() == 1, "footer linked from furniture");

  lighton::docv1::Document empty;
  expect(!lighton::parse_lightonocr_markdown("   ", 1, 0, 0, &empty, &error),
         "empty reply is an error");
  try {
    expect(!lighton::parse_lightonocr_markdown(
        "![text](999999999999999999999999,0,1000,1000)\ntext", 1, 1000, 1000,
        &document, &error), "overflow coordinates rejected");
  } catch (...) { expect(false, "coordinate overflow must not throw"); }
  expect(lighton::parse_lightonocr_markdown(
      "![image](0,0,1000,1000)\nA red boat.", 1, 1000, 1000, &document, &error), error);
  expect(document.pictures_size() == 1 && document.texts_size() == 1,
         "picture description retained");
  if (document.texts_size() == 1) {
    expect(document.texts(0).text().base().text() == "A red boat.", "description text");
    expect(document.texts(0).text().base().parent().ref() == "#/pictures/0", "description parent");
    expect(document.pictures(0).captions_size() == 1, "caption reference");
  }
  expect(!lighton::parse_lightonocr_markdown(
      "![text](" + std::string(200000, '9') + ",0,1000,1000) text",
      1, 1000, 1000, &document, &error), "long coordinate rejected safely");
  expect(lighton::parse_lightonocr_markdown(
      "![" + std::string(200000, 'a') + " malformed marker",
      1, 1000, 1000, &document, &error), "malformed marker remains text");
  expect(!lighton::parse_lightonocr_markdown(std::string(1024 * 1024 + 1, 'x'),
      1, 1000, 1000, &document, &error), "output byte cap enforced");
  if (failures != 0) return 1;
  std::cout << "markdown_test ok\n";
  return 0;
}
